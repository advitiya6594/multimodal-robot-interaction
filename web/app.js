const element = (id) => document.getElementById(id);
let turns = [];
let currentTurn = 0;
let socket;
let busy = false;
let replaying = false;
let pendingRequest = null;
let lastState = null;

function showStatus(text, isError = false) {
  element('status').textContent = text;
  element('status').className = isError ? 'status error' : 'status';
}

function updateControls(isBusy) {
  busy = isBusy;
  for (const id of ['run', 'replay', 'split', 'dialogue', 'mode', 'transcript']) {
    element(id).disabled = busy || replaying;
  }
  document.querySelectorAll('#turns button').forEach((button) => {
    button.disabled = busy || replaying;
  });
  element('stop').disabled = !replaying;
}

function finishRequest(error, result) {
  if (pendingRequest) {
    clearTimeout(pendingRequest.timeout);
    if (error) pendingRequest.reject(error);
    else pendingRequest.resolve(result);
    pendingRequest = null;
  }
  updateControls(false);
}

function renderState(state) {
  element('emotion').textContent = state.emotion;
  element('latency').textContent = `${state.latency_ms} ms compute`;
  element('uncertainty').textContent = state.uncertain
    ? 'Low model confidence · asking for clarification'
    : 'Uncalibrated model scores';
  element('reply').textContent = '';
  element('scores').replaceChildren();
  const rankedScores = Object.entries(state.scores).sort((a, b) => b[1] - a[1]);
  for (const [label, score] of rankedScores) {
    const row = document.createElement('div');
    row.className = 'score';
    const name = document.createElement('span');
    name.textContent = label;
    const bar = document.createElement('div');
    bar.className = 'bar';
    const fill = document.createElement('div');
    fill.className = 'fill';
    fill.style.width = `${score * 100}%`;
    bar.append(fill);
    const value = document.createElement('span');
    value.textContent = `${(score * 100).toFixed(0)}%`;
    row.append(name, bar, value);
    element('scores').append(row);
  }
  const colors = { warm: '#f1cf85', gold: '#efbc4e', white: '#cedacc' };
  element('lamp').style.background = colors[state.behavior.light];
  element('lamp').className = `lamp ${state.behavior.motion}`;
  element('tags').textContent = `Light · ${state.behavior.light} / Tempo · ${state.behavior.tempo} / Motion · ${state.behavior.motion}`;
  element('json').textContent = JSON.stringify(state, null, 2);
}

function connect() {
  return new Promise((resolve, reject) => {
    const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(`${protocol}://${location.host}/ws`);
    socket.onopen = resolve;
    socket.onerror = () => reject(new Error('Cannot connect to local server'));
    socket.onclose = () => {
      finishRequest(new Error('Connection closed'));
    };
    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === 'error') {
        showStatus(message.message, true);
        finishRequest(new Error(message.message));
      }
      if (message.type === 'state') {
        lastState = message;
        renderState(message);
      }
      if (message.type === 'response_chunk') {
        element('reply').textContent += message.text;
      }
      if (message.type === 'done') {
        lastState.response = message.response;
        element('reply').textContent = message.response;
        element('json').textContent = JSON.stringify(lastState, null, 2);
        const entry = document.createElement('div');
        entry.className = 'entry';
        const modelName = element('mode').selectedOptions[0].text;
        entry.textContent = `${lastState.turn_id} · ${modelName} · ${lastState.emotion} — ${message.response}`;
        element('history').prepend(entry);
        showStatus('Turn complete');
        finishRequest(null, lastState);
      }
    };
  });
}

function chooseTurn(index) {
  currentTurn = index;
  const turn = turns[index];
  element('transcript').value = turn.text;
  element('speaker').textContent = turn.speaker;
  element('truth').textContent = `MELD annotation · ${turn.label}`;
  element('position').textContent = `Turn ${index + 1} of ${turns.length}`;
  element('video').src = `/media/${element('split').value}/${turn.id}`;
  document.querySelectorAll('#turns button').forEach((button, i) => {
    button.classList.toggle('active', i === index);
  });
  showStatus(turn.has_audio ? 'Ready' : 'Audio unavailable for this turn');
}

async function loadTurns() {
  const split = element('split').value;
  const dialogue = element('dialogue').value;
  const response = await fetch(`/api/turns?split=${split}&dialogue=${dialogue}`);
  if (!response.ok) throw new Error('Could not load dialogue turns');
  turns = await response.json();
  element('turns').replaceChildren();
  turns.forEach((turn, index) => {
    const button = document.createElement('button');
    button.textContent = String(index + 1);
    button.onclick = () => chooseTurn(index);
    element('turns').append(button);
  });
  if (turns.length) chooseTurn(0);
}

async function loadDialogues() {
  const response = await fetch(`/api/dialogues?split=${element('split').value}`);
  if (!response.ok) throw new Error('Could not load dialogue list');
  const ids = await response.json();
  element('dialogue').replaceChildren();
  for (const id of ids) {
    const option = document.createElement('option');
    option.value = id;
    option.textContent = String(id);
    element('dialogue').append(option);
  }
  await loadTurns();
}

async function processTurn() {
  updateControls(true);
  try {
    if (!socket || socket.readyState !== WebSocket.OPEN) await connect();
    showStatus('Processing locally…');
    return await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        finishRequest(new Error('Inference timed out. Reconnect and try again.'));
        socket.close();
      }, 30000);
      pendingRequest = { resolve, reject, timeout };
      socket.send(JSON.stringify({
        split: element('split').value,
        clip_id: turns[currentTurn].id,
        mode: element('mode').value,
        transcript: element('transcript').value,
      }));
    });
  } finally {
    updateControls(false);
  }
}

function playCurrentClip() {
  return new Promise((resolve, reject) => {
    const video = element('video');
    function cleanUp() {
      video.onended = null;
      video.onerror = null;
      video.removeEventListener('replaystop', stop);
    }
    function stop() {
      cleanUp();
      resolve();
    }
    video.onended = stop;
    video.onerror = () => {
      cleanUp();
      reject(new Error('Clip could not be played'));
    };
    video.addEventListener('replaystop', stop, { once: true });
    video.play().catch((error) => {
      cleanUp();
      reject(error);
    });
  });
}

async function replayConversation() {
  replaying = true;
  updateControls(false);
  try {
    for (let index = 0; index < turns.length && replaying; index += 1) {
      chooseTurn(index);
      await playCurrentClip();
      if (replaying) await processTurn();
    }
  } catch (error) {
    showStatus(error.message, true);
  } finally {
    replaying = false;
    updateControls(false);
  }
}

element('run').onclick = () => processTurn().catch((error) => showStatus(error.message, true));
element('replay').onclick = replayConversation;
element('split').onchange = () => loadDialogues().catch((error) => showStatus(error.message, true));
element('dialogue').onchange = () => loadTurns().catch((error) => showStatus(error.message, true));
element('stop').onclick = () => {
  replaying = false;
  element('video').pause();
  element('video').dispatchEvent(new Event('replaystop'));
  updateControls(busy);
  showStatus('Replay stopped');
};

updateControls(true);
loadDialogues()
  .catch((error) => showStatus(error.message, true))
  .finally(() => updateControls(false));
