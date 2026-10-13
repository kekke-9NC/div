"""Measure real RTSP recording, backlog catch-up and continuous detection.

The RTSP publisher is a loopback server sending copied recordings at 1x.
The original recorder and production detector are used.
Only observation, an initial analysis barrier, and recorder-only cancellation
are added. Installed application files/settings and original recordings are
read-only; every recording and generated artifact belongs to a fresh root.
"""
import contextlib
from datetime import datetime,timedelta
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlsplit


class RedactedLog:
    def __init__(self,handle):
        self.handle,self.buffer,self.lock = handle,'',threading.RLock()

    def write(self,value):
        with self.lock:
            self.buffer += value
            while '\n' in self.buffer:
                line,self.buffer = self.buffer.split('\n',1)
                self.handle.write(re.sub(r'rtsp[s]?://[^\s]+','rtsp://[redacted]',line)+'\n')
                self.handle.flush()
        return len(value)

    def flush(self):
        self.handle.flush()


class LiveMeter:
    def __init__(self):
        self.lock = threading.RLock()
        self.started = time.monotonic()
        self.first_received = self.first_clock = self.received_edge = None
        self.analysis_started = self.capture_stopped = None
        self.processed_edge = None
        self.chunks,self.consumed,self.windows,self.samples,self.errors = {},set(),[],[],[]
        self.sessions = set()

    def receive(self,path,data):
        with self.lock:
            if str(path) in self.chunks:
                return
            now = time.monotonic()
            start = datetime.fromisoformat(data['start_time'])
            end = start+timedelta(seconds=data['frame_count']/data['fps'])
            if self.first_received is None:
                self.first_received,self.first_clock = now,start
            self.received_edge = max(end,self.received_edge or end)
            self.chunks[str(path)] = dict(data,observed_seconds=now-self.started)
            self.sessions.add(str(path.parent))

    def process(self,capture,end_frame,final):
        with self.lock:
            edge = capture.started_at+timedelta(seconds=(end_frame+1)/capture.fps)
            self.processed_edge = max(edge,self.processed_edge or edge)
            self.windows.append(dict(monotonic_seconds=time.monotonic()-self.started,
                media_end=edge.isoformat(),end_frame=end_frame,final=final))

    def sample(self,queue_size):
        with self.lock:
            now = time.monotonic()
            received = ((self.received_edge-self.first_clock).total_seconds()
                        if self.received_edge is not None else 0.)
            processed = ((self.processed_edge-self.first_clock).total_seconds()
                         if self.processed_edge is not None else 0.)
            result = dict(elapsed_seconds=now-self.started,
                live_seconds=None if self.first_received is None else now-self.first_received,
                analysis_seconds=None if self.analysis_started is None else now-self.analysis_started,
                received_media_seconds=received,processed_media_seconds=processed,
                lag_seconds=max(0.,(self.received_edge-(self.processed_edge or self.first_clock)).total_seconds())
                    if self.received_edge is not None else 0.,queue_chunks=queue_size,
                received_chunks=len(self.chunks),consumed_chunks=len(self.consumed),
                capture_stopped=self.capture_stopped is not None)
            self.samples.append(result)
            return result


def summarize_timeline(samples,backlog_seconds,healthy_lag=3.):
    active = [row for row in samples if row['analysis_seconds'] is not None and not row['capture_stopped']]
    caught = next((row for row in active if row['analysis_seconds']>0 and row['lag_seconds']<=healthy_lag),None)
    catch_seconds = None if caught is None else caught['analysis_seconds']
    speed = None if caught is None else caught['processed_media_seconds']/catch_seconds
    steady = [row for row in active if caught is not None and row['analysis_seconds']>=catch_seconds+5]
    lags = sorted(row['lag_seconds'] for row in steady)
    return dict(initial_backlog_delay_seconds=backlog_seconds,healthy_lag_seconds=healthy_lag,
                catch_up_seconds=catch_seconds,catch_up_media_seconds_per_wall_second=speed,
                caught_up_while_receiving=caught is not None,
                steady_samples=len(steady),steady_max_lag_seconds=max(lags,default=None),
                steady_p95_lag_seconds=lags[min(len(lags)-1,int(len(lags)*.95))] if lags else None,
                final_receiving_lag_seconds=active[-1]['lag_seconds'] if active else None)


def run_live(args,settings,options,noise,correction,mask):
    import cv2
    import config
    import file_utils
    import rtsp_live
    import rtsp_schedule_products as products
    import video_processing
    root = args.output/'rtsp'
    root.mkdir()
    config.RTSP_SAVE_ROOT = str(root)
    url = args.stream_url
    if not isinstance(url,str) or urlsplit(url).scheme not in ('rtsp','rtsps'):
        raise ValueError('A configured RTSP stream is required')
    if urlsplit(url).hostname not in ('127.0.0.1','localhost','::1') or urlsplit(url).username:
        raise ValueError('Verification uses an unauthenticated loopback test stream only')
    if noise.get('enabled') or noise.get('temporal_mean_frames'):
        raise ValueError('Live throughput harness currently expects the saved original-frame workflow')
    meter = LiveMeter()
    cancel,record_cancel,monitor_stop = threading.Event(),threading.Event(),threading.Event()
    runners,finished,results = [],[],[]
    original_runner,original_timing = rtsp_live.RealtimeRunner,rtsp_live.timing
    original_ack,original_recorder = rtsp_live.SegmentCapture.acknowledge,file_utils.save_rtsp_video_segments
    original_index = video_processing._SavedEventIndex
    def observed_timing(path):
        data = original_timing(path)
        path = Path(path)
        if path.is_relative_to(root/'.rtsp_live') and path.name.startswith('chunk_'):
            meter.receive(path,data)
        return data
    def acknowledged(capture,end_frame,final=False):
        original_ack(capture,end_frame,final)
        meter.process(capture,end_frame,final)
    class ObservedIndex(original_index):
        def append(self,item):
            super().append(item)
            results.append(dict(item))
    class ObservedRunner(original_runner):
        def __init__(self,save_root,analyze,*values,**kwargs):
            def measured(capture,evidence):
                release = (meter.first_received or time.monotonic())+args.backlog_seconds
                while not cancel.is_set() and time.monotonic()<release:
                    cancel.wait(min(.1,release-time.monotonic()))
                with meter.lock:
                    if meter.analysis_started is None:
                        meter.analysis_started = time.monotonic()
                try:
                    result = analyze(capture,evidence)
                    if not capture.eof:
                        meter.errors.append('Detector returned before stream EOF')
                    finished.append(str(capture.first_path.parent))
                    return result
                except Exception as exc:
                    meter.errors.append(type(exc).__name__+': '+str(exc))
                    raise
            super().__init__(save_root,measured,*values,**kwargs)
            runners.append(self)
        def _consume(self,path):
            super()._consume(path)
            with meter.lock:
                meter.consumed.add(str(path))
    def record(stream,save_root,segment_duration,shared_cancel,*values,**kwargs):
        # Stop reception independently; the production runner remains alive
        # to finish all windows, event saves and final acknowledgements.
        try:
            original_recorder(stream,save_root,segment_duration,record_cancel,*values,**kwargs)
        finally:
            meter.capture_stopped = time.monotonic()
        for runner in runners:
            runner.flush_requested.set()
        deadline = time.monotonic()+min(180,args.timeout)
        while not cancel.is_set() and time.monotonic()<deadline:
            with meter.lock:
                all_consumed = bool(meter.chunks) and set(meter.chunks)<=meter.consumed
            raw = list((root/'.rtsp_live').rglob('chunk_*.mp4'))
            if all_consumed and finished and not raw and all(r.queue.empty() for r in runners):
                return
            time.sleep(.1)
        meter.errors.append('Capture stopped but detection/archive drain did not complete')
    def monitor():
        while not monitor_stop.wait(1):
            sample = meter.sample(sum(r.queue.qsize() for r in runners))
            rtsp_live.write_json(args.output/'progress.json',sample)
            if meter.first_received is not None and time.monotonic()-meter.first_received>=args.live_seconds:
                record_cancel.set()
            if (meter.first_received is None and time.monotonic()-meter.started>45) or time.monotonic()-meter.started>args.timeout:
                meter.errors.append('Live input/verification deadline expired')
                record_cancel.set()
                cancel.set()
    rtsp_live.RealtimeRunner,rtsp_live.timing = ObservedRunner,observed_timing
    rtsp_live.SegmentCapture.acknowledge,file_utils.save_rtsp_video_segments = acknowledged,record
    video_processing._SavedEventIndex = ObservedIndex
    watcher = threading.Thread(target=monitor,daemon=True,name='LiveVerificationMonitor')
    clock = tuple(int(settings.get(key,fallback)) for key,fallback in
                  (('rtsp_start_hour',17),('rtsp_start_minute',0),('rtsp_end_hour',5),('rtsp_end_minute',0)))
    product_options = products.processing_options(settings.get('rtsp_schedule_products',{}))
    try:
        with (args.output/'pipeline.log').open('w') as handle,contextlib.redirect_stdout(RedactedLog(handle)):
            watcher.start()
            file_utils.rtsp_save_and_process_thread_target(
                rtsp_url=url,save_root=str(root),segment_duration=config.RTSP_SEGMENT_DURATION,
                scan_interval=config.RTSP_SCAN_INTERVAL,progress_callback=lambda _value:None,mask=mask,
                meteor_save_path=str(args.output/'meteor'),not_meteor_save_path=str(args.output/'not_meteor'),
                cancel_flag=cancel,save_options=options,interval=float(settings.get('interval',1)),
                duration=float(settings.get('duration',1)),summary_video_config=settings.get('summary_video_config'),
                time_limit_enabled=bool(settings.get('rtsp_time_limit_enabled',True)),
                start_hour=clock[0],start_minute=clock[1],end_hour=clock[2],end_minute=clock[3],
                max_workers=int(settings.get('concurrency',4)),dark_frame=correction,
                notify_on_detection=False,noise_twin_options=noise,rtsp_fps=config.RTSP_FPS,
                schedule_product_options=product_options)
    finally:
        record_cancel.set()
        cancel.set()
        monitor_stop.set()
        watcher.join(timeout=5)
        products.stop_background_products()
        rtsp_live.RealtimeRunner,rtsp_live.timing = original_runner,original_timing
        rtsp_live.SegmentCapture.acknowledge,file_utils.save_rtsp_video_segments = original_ack,original_recorder
        video_processing._SavedEventIndex = original_index
    final = meter.sample(sum(r.queue.qsize() for r in runners))
    (args.output/'timeline.json').write_text(json.dumps(meter.samples,indent=2))
    (args.output/'window_progress.json').write_text(json.dumps(meter.windows,indent=2))
    (args.output/'received_chunks.json').write_text(json.dumps(meter.chunks,indent=2))
    threads_alive = [thread.name for runner in runners for thread in runner.threads if thread.is_alive()]
    all_consumed = bool(meter.chunks) and set(meter.chunks)<=meter.consumed
    archive_paths = sorted(root.glob('[0-9]*/[0-9]*/*_minute_*.mp4'))
    archived_frames = sum(int(rtsp_live.timing(path)['frame_count']) for path in archive_paths)
    input_frames = sum(int(data['frame_count']) for data in meter.chunks.values())
    sessions = len(meter.sessions)
    metadata = sorted(meter.chunks.items(),key=lambda item:item[1]['start_time'])
    gaps = []
    for (before,a),(after,b) in zip(metadata,metadata[1:]):
        expected = datetime.fromisoformat(a['start_time'])+timedelta(seconds=a['frame_count']/a['fps'])
        gap = (datetime.fromisoformat(b['start_time'])-expected).total_seconds()
        if abs(gap)>max(.1,2/a['fps']) or a['fps']!=b['fps'] or a['size']!=b['size']:
            gaps.append(dict(before=before,after=after,gap_seconds=gap))
    timing = summarize_timeline(meter.samples,args.backlog_seconds)
    receiving_wall = (meter.capture_stopped-meter.first_received if meter.capture_stopped is not None and meter.first_received is not None else None)
    input_rate = final['received_media_seconds']/receiving_wall if receiving_wall else None
    recorded_frames_ok = archived_frames==input_frames
    valid = (not meter.errors and not threads_alive and all_consumed and recorded_frames_ok
             and sessions==1 and not gaps and input_rate is not None and .95<=input_rate<=1.05)
    keeps_up = (valid and timing['caught_up_while_receiving'] and timing['steady_samples']>=30
                and timing['steady_p95_lag_seconds']<=5 and timing['final_receiving_lag_seconds']<=5)
    jobs = [json.loads(path.read_text()) for path in (root/'.schedule_jobs').glob('*/job.json')]
    result = dict(workflow='live_rtsp',stream_kind='loopback',
        recorder_backend='ffmpeg' if config.RTSP_USE_NVIDIA_HWACCEL and correction is None else 'opencv',
        captured_media_seconds=final['received_media_seconds'],receiving_wall_seconds=receiving_wall,
        received_media_seconds_per_wall_second=input_rate,input_frames=input_frames,archives=len(archive_paths),
        archived_frames=archived_frames,recorded_frames_ok=recorded_frames_ok,
        all_input_consumed=all_consumed,all_runner_threads_stopped=not threads_alive,
        threads_alive=threads_alive,sessions=sessions,timing_gaps=gaps,errors=meter.errors,
        media_enabled=True,scheduled_products_enabled=product_options,
        scheduled_job_states=[dict(id=job['id'],state=job['state'],end=job['end']) for job in jobs],
        overnight_product_finalization_verified=False,verification_valid=valid,
        realtime_keeps_up_verified=keeps_up,**timing)
    return result,results
