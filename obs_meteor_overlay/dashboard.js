'use strict';
const byId = id => document.getElementById(id);
const replay = byId('replay');
const jstClock = new Intl.DateTimeFormat('ja-JP', {timeZone: 'Asia/Tokyo', hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23'});
const jstDate = new Intl.DateTimeFormat('ja-JP', {timeZone: 'Asia/Tokyo', year: 'numeric', month: '2-digit', day: '2-digit'});
const jstMinute = new Intl.DateTimeFormat('ja-JP', {timeZone: 'Asia/Tokyo', hour: '2-digit', minute: '2-digit', hourCycle: 'h23'});
let displayedEvent = null;
let displayedEpoch = null;
let displayedReplayEpoch = null;
let imageKey = null;
let pendingKey = null;
let focusTimer = null;
let metadataBusy = false;
let statisticsBusy = false;
let statsUpdatedEpoch = null;
let skyBusy = false;
let skyGeneration = null;
let skyUpdatedEpoch = null;
let historyBusy = false;
let historyKey = null;
let historyData = null;
let historyAnimation = null;
let boardTimer = null;
let historyActive = false;
let renderedHistoryEvents = [];

function updateClock() {
    const now = new Date();
    if (skyUpdatedEpoch === null || now.getTime() / 1000 - skyUpdatedEpoch > 20) byId('sky').hidden = true;
    byId('current-clock').textContent = jstClock.format(now);
    byId('current-date').textContent = jstDate.format(now);
    if (displayedReplayEpoch !== null) {
        const seconds = Math.max(0, now.getTime() / 1000 - displayedReplayEpoch);
        byId('elapsed').textContent = seconds < 60 ? 'たった今' : seconds < 3600 ? `${Math.floor(seconds / 60)}分前` : seconds < 86400 ? `${Math.floor(seconds / 3600)}時間前` : `${Math.floor(seconds / 86400)}日前`;
    }
    if (statsUpdatedEpoch !== null) {
        byId('stats-state').textContent = now.getTime() / 1000 - statsUpdatedEpoch > 90 ? '更新が遅れています' : '訂正反映・未確認は自動判定';
    }
}

async function refreshSky() {
    if (skyBusy) return;
    skyBusy = true;
    try {
        const data = await localJSON('sky_overlay.json');
        const fresh = Number.isFinite(data.updated_at_epoch) && Math.abs(Date.now() / 1000 - data.updated_at_epoch) < 20;
        if (!data.enabled || !fresh || data.width !== 1920 || data.height !== 1080) {
            byId('sky').hidden = true;
            skyUpdatedEpoch = null;
            return;
        }
        if (data.generation !== skyGeneration) {
            const source = await loadImage('sky_overlay.png', data.generation);
            if (Date.now() / 1000 - data.updated_at_epoch > 20) return;
            byId('sky-lines').src = source;
            skyGeneration = data.generation;
        }
        skyUpdatedEpoch = data.updated_at_epoch;
        byId('sky').hidden = false;
    } catch (_) {
        // The clock hides an expired layer if its renderer stops.
    } finally {
        skyBusy = false;
    }
}

async function localJSON(name) {
    const url = new URL(name, document.location.href);
    url.searchParams.set('v', Date.now());
    const response = await fetch(url.href, {cache: 'no-store'});
    if (!response.ok) throw new Error('Metadata unavailable');
    return response.json();
}

function loadImage(name, key) {
    return new Promise((resolve, reject) => {
        const url = new URL(name, document.location.href);
        url.searchParams.set('v', key);
        const image = new Image();
        image.onload = () => resolve(image.src);
        image.onerror = reject;
        image.src = url.href;
    });
}

async function refreshMetadata() {
    if (metadataBusy) return;
    metadataBusy = true;
    try {
        const data = await localJSON('latest_meteor.json');
        const nightly = Number(data.tonight_count);
        if (data.tonight_count != null && Number.isFinite(nightly)) byId('count').textContent = nightly;
        byId('window').textContent = data.tonight_window || '18:00–翌06:00';
        if (!data.event || data.detected_at_epoch == null) {
            clearTimeout(focusTimer);
            replay.classList.remove('ready', 'focus', 'located');
            byId('meteor').removeAttribute('src');
            byId('zoom').removeAttribute('src');
            byId('last-date').textContent = '--/--';
            byId('last-clock').textContent = '--:--:--';
            byId('elapsed').textContent = '--';
            byId('replay-state').textContent = '自動リプレイ';
            displayedEvent = displayedEpoch = displayedReplayEpoch = imageKey = null;
            byId('visible-window').textContent = '';
            return;
        }
        const epoch = Number(data.detected_at_epoch);
        if (!Number.isFinite(epoch)) return;
        const key = `${data.event}:${data.mask_revision || ''}:${JSON.stringify(data.zoom_region || null)}`;
        // Reload only when the event or mask changes; polling must not restart GIFs.
        if (key === imageKey) {
            showReplayTime(data);
            return;
        }
        if (key === pendingKey) return;
        pendingKey = key;
        const [full, zoom] = await Promise.all([
            loadImage('latest_meteor.gif', key),
            loadImage('latest_meteor_zoom.gif', key),
        ]);
        const newDetection = displayedEvent !== null && data.event !== displayedEvent && epoch > displayedEpoch;
        byId('meteor').src = full;
        byId('zoom').src = zoom;
        displayedEvent = data.event;
        displayedEpoch = epoch;
        imageKey = key;
        showReplayTime(data);
        replay.classList.add('ready');
        showZoomRegion(data.zoom_region);
        if (newDetection) {
            clearTimeout(focusTimer);
            replay.classList.add('focus');
            byId('replay-state').textContent = '新しい検出 · リプレイ';
            focusTimer = setTimeout(() => {
                replay.classList.remove('focus');
                byId('replay-state').textContent = '自動リプレイ';
            }, 12000);
        }
        updateClock();
    } catch (_) {
        // Keep the last complete replay while local outputs are being replaced.
    } finally {
        pendingKey = null;
        metadataBusy = false;
    }
}

function showReplayTime(data) {
    const epoch = Number(data.replay_at_epoch ?? data.detected_at_epoch);
    if (!Number.isFinite(epoch)) return;
    displayedReplayEpoch = epoch;
    byId('last-date').textContent = jstDate.format(new Date(epoch * 1000));
    byId('last-clock').textContent = jstClock.format(new Date(epoch * 1000));
    const aligned = data.timestamp_basis === 'obs_frame_match';
    byId('time-basis').textContent = aligned ? '配信映像の時刻 · JST' : data.timestamp_basis === 'recording_frame' ? '録画時刻 · 配信照合待ち' : '検出記録の時刻';
    const validRange = Number.isFinite(data.visible_start_epoch) && Number.isFinite(data.visible_end_epoch);
    byId('visible-window').textContent = validRange ? `流星 ${jstClock.format(new Date(data.visible_start_epoch * 1000))}–${jstClock.format(new Date(data.visible_end_epoch * 1000))}` : '';
}

function renderHistory(data) {
    renderedHistoryEvents = data.events.map(event => event.event);
    const track = byId('history-track');
    const cards = data.events.map((event, index) => {
        const card = document.createElement('div');
        card.className = 'history-card';
        const image = document.createElement('img');
        image.alt = `流星 ${index + 1} の拡大映像`;
        if (event.image && /^recent_cache\/[a-f0-9]{20}\.gif$/.test(event.image)) image.src = new URL(event.image, document.location.href).href;
        const info = document.createElement('div');
        info.className = 'history-info';
        const rank = document.createElement('span');
        rank.textContent = index === 0 ? '最新' : `${index + 1}番目`;
        const clock = document.createElement('time');
        const epoch = Number(event.replay_at_epoch ?? event.detected_at_epoch);
        clock.textContent = Number.isFinite(epoch) ? jstClock.format(new Date(epoch * 1000)) : '--:--:--';
        const date = document.createElement('span');
        date.textContent = Number.isFinite(epoch) ? jstDate.format(new Date(epoch * 1000)).slice(5) : '';
        const basis = document.createElement('span');
        basis.textContent = !event.image ? '映像準備中' : event.timestamp_basis === 'obs_frame_match' ? '配信時刻' : '録画時刻';
        info.append(rank, clock, date, basis);
        card.append(image, info);
        return card;
    });
    if (!cards.length) {
        const empty = document.createElement('span');
        empty.className = 'history-empty';
        empty.textContent = '流星の検出を待っています';
        cards.push(empty);
    }
    track.replaceChildren(...cards);
    byId('history-title').textContent = `直近${data.events.length || 10}件の流星`;
    byId('history-state').textContent = '新しい順 · 拡大リプレイ · 訂正反映';
}

function switchBoard() {
    clearTimeout(boardTimer);
    historyActive = !historyActive && Boolean(historyData && historyData.events.length);
    byId('graph-page').classList.toggle('active', !historyActive);
    byId('history-page').classList.toggle('active', historyActive);
    byId('graph-page').setAttribute('aria-hidden', String(historyActive));
    byId('history-page').setAttribute('aria-hidden', String(!historyActive));
    if (historyAnimation) historyAnimation.cancel();
    if (historyActive) {
        renderHistory(historyData);
        const distance = Math.max(0, byId('history-track').scrollWidth - byId('history-viewport').clientWidth);
        const duration = Math.max(18000, distance / 24 * 1000 + 10000);
        historyAnimation = byId('history-track').animate([
            {transform: 'translateX(0)', offset: 0},
            {transform: 'translateX(0)', offset: 4000 / duration},
            {transform: `translateX(-${distance}px)`, offset: 1 - 6000 / duration},
            {transform: `translateX(-${distance}px)`, offset: 1},
        ], {duration, easing: 'linear', fill: 'forwards'});
        boardTimer = setTimeout(switchBoard, duration);
    } else {
        boardTimer = setTimeout(switchBoard, 24000);
    }
}

async function refreshHistory() {
    if (historyBusy) return;
    historyBusy = true;
    try {
        const data = await localJSON('recent_meteors.json');
        if (!Array.isArray(data.events) || data.events.length > 10) return;
        if (!Number.isFinite(data.updated_at_epoch) || Date.now() / 1000 - data.updated_at_epoch > 90) {
            historyData = null;
            if (historyActive) switchBoard();
            byId('board-next').textContent = '流星一覧の更新待ち';
            return;
        }
        byId('board-next').textContent = data.events.length ? `次：直近${data.events.length}件の流星` : '流星の検出を待っています';
        const key = JSON.stringify([data.mask_revision, data.events.map(event => [event.event, event.image]), data.excluded_events]);
        if (key !== historyKey) {
            // Newly detected items join on the next cycle, without interrupting
            // scrolling. Removed reviews/masks hide the affected list at once.
            const invalidated = historyData && (data.mask_revision !== historyData.mask_revision || renderedHistoryEvents.some(event => (data.excluded_events || []).includes(event)));
            historyData = data;
            historyKey = key;
            if (historyActive && invalidated) switchBoard();
        } else {
            historyData = data;
        }
    } catch (_) {
        byId('board-next').textContent = '流星一覧を準備しています';
        if (historyData && Date.now() / 1000 - historyData.updated_at_epoch > 90 && historyActive) switchBoard();
    } finally {
        historyBusy = false;
    }
}

function showZoomRegion(region) {
    const values = region && [region.x, region.y, region.width, region.height];
    const valid = values && values.every(Number.isFinite) && region.width > 0 && region.height > 0 && region.x >= 0 && region.y >= 0 && region.x + region.width <= 320.01 && region.y + region.height <= 180.01;
    replay.classList.toggle('located', Boolean(valid));
    byId('zoom-label').textContent = valid ? '水色の範囲 · 拡大' : '流星拡大';
    if (!valid) return;
    const {x, y, width: w, height: h} = region;
    const arm = Math.min(9, w / 3, h / 3);
    // Corners leave the meteor itself unobscured while marking the exact crop.
    byId('zoom-brackets').setAttribute('d', [
        `M${x},${y + arm} V${y} H${x + arm}`,
        `M${x + w - arm},${y} H${x + w} V${y + arm}`,
        `M${x + w},${y + h - arm} V${y + h} H${x + w - arm}`,
        `M${x + arm},${y + h} H${x} V${y + h - arm}`,
    ].join(' '));
}

function drawGraph(bins) {
    const svg = byId('graph');
    const ns = 'http://www.w3.org/2000/svg';
    const elements = [];
    const add = (tag, attrs, text) => {
        const el = document.createElementNS(ns, tag);
        for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
        if (text !== undefined) el.textContent = text;
        elements.push(el);
    };
    const max = Math.max(1, ...bins);
    // Fixed zero baseline; bar heights use the largest of the six counts.
    add('line', {x1: 12, x2: 988, y1: 97, y2: 97, stroke: '#425169'});
    bins.forEach((value, index) => {
        const x = 26 + index * 162;
        const h = value / max * 65;
        add('rect', {x, y: 97 - h, width: 126, height: h, rx: 3, class: 'bar'});
        add('text', {x: x + 63, y: 89 - h, 'text-anchor': 'middle', class: 'bar-value'}, value);
        add('text', {x: x + 63, y: 122, 'text-anchor': 'middle'}, `${60 - index * 10}–${50 - index * 10}分前`);
    });
    svg.replaceChildren(...elements);
    svg.setAttribute('aria-label', `直近1時間、古い順に10分ごとの検出数：${bins.join('、')}件`);
}

async function refreshStatistics() {
    if (statisticsBusy) return;
    statisticsBusy = true;
    try {
        const data = await localJSON('hourly_stats.json');
        if (!Array.isArray(data.bins) || data.bins.length !== 6 || !data.bins.every(n => Number.isInteger(n) && n >= 0)) return;
        const start = Number(data.window_start_epoch);
        const end = Number(data.window_end_epoch);
        if (!Number.isFinite(start) || !Number.isFinite(end)) return;
        byId('hour-count').textContent = data.bins.reduce((sum, n) => sum + n, 0);
        byId('hour-window').textContent = `${jstMinute.format(new Date(start * 1000))} – ${jstMinute.format(new Date(end * 1000))}`;
        statsUpdatedEpoch = Number(data.updated_at_epoch);
        drawGraph(data.bins);
        updateClock();
    } catch (_) {
        // Preserve counts instead of showing a false zero on a read failure.
    } finally {
        statisticsBusy = false;
    }
}

updateClock();
refreshMetadata();
refreshStatistics();
refreshSky();
refreshHistory();
boardTimer = setTimeout(switchBoard, 24000);
setInterval(updateClock, 1000);
setInterval(refreshMetadata, 2500);
setInterval(refreshStatistics, 2500);
setInterval(refreshSky, 2500);
setInterval(refreshHistory, 5000);
