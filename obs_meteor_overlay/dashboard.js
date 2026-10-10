'use strict';
const byId = id => document.getElementById(id);
const replay = byId('replay');
const jstClock = new Intl.DateTimeFormat('ja-JP', {timeZone: 'Asia/Tokyo', hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23'});
const jstDate = new Intl.DateTimeFormat('ja-JP', {timeZone: 'Asia/Tokyo', year: 'numeric', month: '2-digit', day: '2-digit'});
const jstMinute = new Intl.DateTimeFormat('ja-JP', {timeZone: 'Asia/Tokyo', hour: '2-digit', minute: '2-digit', hourCycle: 'h23'});
let displayedEvent = null;
let displayedEpoch = null;
let imageKey = null;
let pendingKey = null;
let focusTimer = null;
let metadataBusy = false;
let statisticsBusy = false;
let statsUpdatedEpoch = null;

function updateClock() {
    const now = new Date();
    byId('current-clock').textContent = jstClock.format(now);
    byId('current-date').textContent = jstDate.format(now);
    if (displayedEpoch !== null) {
        const seconds = Math.max(0, now.getTime() / 1000 - displayedEpoch);
        byId('elapsed').textContent = seconds < 60 ? 'たった今' : seconds < 3600 ? `${Math.floor(seconds / 60)}分前` : seconds < 86400 ? `${Math.floor(seconds / 3600)}時間前` : `${Math.floor(seconds / 86400)}日前`;
    }
    if (statsUpdatedEpoch !== null) {
        byId('stats-state').textContent = now.getTime() / 1000 - statsUpdatedEpoch > 90 ? '更新が遅れています' : '保存された検出を集計';
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
        if (!data.event || data.detected_at_epoch == null) return;
        const epoch = Number(data.detected_at_epoch);
        if (!Number.isFinite(epoch)) return;
        const key = `${data.event}:${data.mask_revision || ''}`;
        // Reload only when the event or mask changes; polling must not restart GIFs.
        if (key === imageKey || key === pendingKey) return;
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
        byId('last-date').textContent = jstDate.format(new Date(epoch * 1000));
        byId('last-clock').textContent = jstClock.format(new Date(epoch * 1000));
        replay.classList.add('ready');
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
setInterval(updateClock, 1000);
setInterval(refreshMetadata, 2500);
setInterval(refreshStatistics, 2500);
