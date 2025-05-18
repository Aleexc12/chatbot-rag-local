// static/script.js

document.addEventListener('DOMContentLoaded', () => {
    // Refs a elementos del DOM
    const appContainer     = document.getElementById('app-container');
    const toggleSidebarBtn = document.getElementById('toggle-sidebar-btn');
    const newChatBtn       = document.getElementById('new-chat-btn');
    const chatHistoryList  = document.getElementById('chat-history-list');
    const modelBtn         = document.getElementById('model-btn');
    const modelMenu        = document.getElementById('model-menu');
    const dropZone         = document.getElementById('drop-zone');
    const fileInput        = document.getElementById('file-input');
    const fileList         = document.getElementById('file-list');
    const errorStatus      = document.getElementById('error-status');
    const userInput        = document.getElementById('user-input');
    const sendButton       = document.getElementById('send-button');
    const chatBox          = document.getElementById('chat-box');

    // Estado de la aplicación - Inyectado desde el HTML (Flask)
    // Asegúrate de que estas variables se definen globalmente en el script.
    // Ya están definidas en el HTML dentro de un tag <script>
    // let selectedModel; (definida en HTML)
    // let currentSessionId; (definida en HTML)
    // const availableLLMs; (definida en HTML)

    // --- Helper Functions ---
    function scrollToBottom() {
      const wrapper = document.querySelector('.chat-box-wrapper');
      wrapper.scrollTop = wrapper.scrollHeight;
    }

    function escapeHtml(unsafe) {
        return unsafe
             .replace(/&/g, "&")
             .replace(/</g, "<")
             .replace(/>/g, ">")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#39;");
    }

    function createMessageBubble(text, sender = 'bot', thinking = false, isSystemMessage = false) {
      const outerContainer = document.createElement('div');
      const messageBubble = document.createElement('div');
      
      if (isSystemMessage) {
          messageBubble.classList.add('system-message');
          messageBubble.innerHTML = `<i>${escapeHtml(text)}</i>`;
          chatBox.appendChild(messageBubble);
          scrollToBottom();
          return messageBubble;
      }
      
      outerContainer.className = `message-container ${sender === 'user' ? 'message-user-container' : 'message-bot-container'}`;
      messageBubble.classList.add(sender === 'user' ? 'message-user' : 'message-bot');

      if (thinking) {
        messageBubble.classList.add('thinking');
        messageBubble.innerHTML = `<i>${escapeHtml(text)}</i>`;
      } else {
        messageBubble.innerHTML = escapeHtml(text).replace(/\n/g, '<br>');
      }
      
      outerContainer.appendChild(messageBubble);
      chatBox.appendChild(outerContainer);
      scrollToBottom();
      return messageBubble;
    }

    function addTimestamp(bubbleElement, timestampText) {
      if (!bubbleElement || !timestampText || bubbleElement.classList.contains('system-message')) return;
      let tsElement = bubbleElement.querySelector('.message-timestamp-inline');
      if (!tsElement) {
        tsElement = document.createElement('span');
        tsElement.className = 'message-timestamp-inline';
        bubbleElement.appendChild(tsElement);
      }
      tsElement.textContent = timestampText;
    }

    function addSources(bubbleElement, sourcesArray) {
      if (!bubbleElement || !sourcesArray?.length || bubbleElement.classList.contains('system-message')) return;
      let sourcesDiv = bubbleElement.querySelector('.message-sources');
      if (!sourcesDiv) {
        sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        bubbleElement.appendChild(sourcesDiv);
      }
      
      const items = sourcesArray.map(s => {
        const fn = s.filename || 'unknown';
        const id = s.id ? s.id.substring(0, 8) : 'N/A';
        const disp = fn.length > 20 ? fn.slice(0, 17) + '...' : fn;
        return `${escapeHtml(disp)} (ID:${escapeHtml(id)})`;
      });
      let textContent = items.slice(0, 2).join(', ');
      if (items.length > 2) textContent += `, +${items.length - 2} more`;
      
      sourcesDiv.textContent = `Sources: ${textContent}`;
      sourcesDiv.title = sourcesArray.map(s => `${escapeHtml(s.filename)}(ID:${escapeHtml(s.id || 'N/A')})`).join(', ');
    }

    let errorTimeout;
    function displayError(messageText) {
      clearTimeout(errorTimeout);
      errorStatus.textContent = messageText || 'An error occurred.';
      errorStatus.style.display = 'block';
      errorTimeout = setTimeout(() => errorStatus.style.display = 'none', 5000);
    }

    function setChatInputEnabled(isEnabled) {
      userInput.disabled = !isEnabled;
      sendButton.disabled = !isEnabled;
      userInput.placeholder = isEnabled ? 'Type your message…' : 'Processing…';
      userInput.style.backgroundColor = isEnabled ? '' : '#f8f9fa';
    }

    userInput.addEventListener('input', () => {
      userInput.style.height = 'auto';
      userInput.style.height = userInput.scrollHeight + 'px';
    });

    // --- Chat History Functions ---
    function generateUUID() {
        return ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
            (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
        );
    }

    function startNewChat() {
        // La variable global `currentSessionId` se actualiza aquí.
        // Las variables `selectedModel` y `availableLLMs` ya son globales.
        currentSessionId = generateUUID(); 
        console.log("New chat session started. Client-side ID:", currentSessionId);
        chatBox.innerHTML = '';
        fileList.innerHTML = '';
        errorStatus.style.display = 'none';
        userInput.value = '';
        userInput.style.height = 'auto';
        userInput.focus();
        setChatInputEnabled(true);
        
        document.querySelectorAll('#chat-history-list li.active').forEach(item => item.classList.remove('active'));
        // Opcional: añadir un mensaje de bienvenida para el nuevo chat
        createMessageBubble("New chat started. How can I help you?", "bot", false, true);
    }

    async function loadChatSessions() {
        try {
            const response = await fetch('/get_chat_sessions');
            if (!response.ok) throw new Error(`Failed to load sessions: ${response.status}`);
            const sessions = await response.json();
            renderChatSessions(sessions);
        } catch (error) {
            console.error('Error loading chat sessions:', error);
            chatHistoryList.innerHTML = '<li class="no-history-item">Error loading history.</li>';
        }
    }

    function renderChatSessions(sessions) {
        chatHistoryList.innerHTML = '';
        if (!sessions || sessions.length === 0) {
            const noHistoryItem = document.createElement('li');
            noHistoryItem.textContent = 'No past chats.';
            noHistoryItem.classList.add('no-history-item');
            chatHistoryList.appendChild(noHistoryItem);
            return;
        }

        sessions.forEach(session => {
            const listItem = document.createElement('li');
            listItem.dataset.sessionId = session.session_id;
            listItem.textContent = session.display_name;
            listItem.title = `Chat ID: ${session.session_id}\nLast activity: ${new Date(session.last_activity).toLocaleString()}`;

            if (session.session_id === currentSessionId) {
                listItem.classList.add('active');
            }

            listItem.addEventListener('click', () => {
                if (currentSessionId === session.session_id && chatBox.hasChildNodes() && !chatBox.querySelector('.system-message i:contains("Loading messages")')) {
                    return; 
                }
                currentSessionId = session.session_id;
                console.log("Switched to session:", currentSessionId);
                
                document.querySelectorAll('#chat-history-list li.active').forEach(item => item.classList.remove('active'));
                listItem.classList.add('active');
                loadAndDisplayMessagesForSession(currentSessionId);
            });
            chatHistoryList.appendChild(listItem);
        });
    }

    async function loadAndDisplayMessagesForSession(sessionId) {
        chatBox.innerHTML = '<div class="system-message"><i>Loading messages...</i></div>';
        try {
            const response = await fetch(`/get_messages_for_session/${sessionId}`);
            if (!response.ok) throw new Error(`Failed to load messages: ${response.status}`);
            const messages = await response.json();
            chatBox.innerHTML = '';

            if (messages.length === 0) {
                createMessageBubble("This chat seems to be empty.", "bot", false, true);
            } else {
                messages.forEach(msg => {
                    const isUser = msg.role === 'user';
                    const isSystem = msg.role === 'system';
                    const bubble = createMessageBubble(msg.content, msg.role, false, isSystem);
                    if (!isSystem) {
                        addTimestamp(bubble, new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
                    }
                });
            }
            scrollToBottom();
        } catch (error) {
            console.error('Error loading messages for session:', sessionId, error);
            chatBox.innerHTML = `<div class="system-message error"><i>Error loading messages: ${escapeHtml(error.message)}</i></div>`;
        }
    }
    
    if (newChatBtn) {
        newChatBtn.addEventListener('click', startNewChat);
    }

    // --- Event Listeners (Drag & Drop, Sidebar, Model Dropdown, Ctrl+Enter) ---
    function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }
    function highlight() { dropZone.classList.add('drag-over'); }
    function unhighlight() { dropZone.classList.remove('drag-over'); }
    function handleDrop(e) { preventDefaults(e); unhighlight(); handleFiles(e.dataTransfer.files); }

    dropZone.addEventListener('click', () => fileInput.click());
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev => {
      dropZone.addEventListener(ev, preventDefaults);
      document.body.addEventListener(ev, preventDefaults);
    });
    ['dragenter', 'dragover'].forEach(ev => dropZone.addEventListener(ev, highlight));
    ['dragleave', 'drop'].forEach(ev => dropZone.addEventListener(ev, unhighlight));
    dropZone.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', () => handleFiles(fileInput.files));

    toggleSidebarBtn.addEventListener('click', () => {
      appContainer.classList.toggle('sidebar-collapsed');
    });

    function populateModelDropdown() {
        modelMenu.innerHTML = '<div class="model-item disabled">Choose LLM…</div>';
        for (const key in availableLLMs) {
            if (availableLLMs.hasOwnProperty(key)) { // Ensure it's an own property
                const llm = availableLLMs[key];
                const item = document.createElement('div');
                item.classList.add('model-item');
                item.dataset.value = key;
                item.textContent = llm.display_name;
                if (key === selectedModel) {
                    item.classList.add('selected');
                    modelBtn.innerHTML = `${escapeHtml(llm.display_name)} <i class="arrow"></i>`;
                }
                item.addEventListener('click', () => {
                    const previouslySelected = modelMenu.querySelector('.model-item.selected');
                    if (previouslySelected) previouslySelected.classList.remove('selected');
                    item.classList.add('selected');
                    
                    selectedModel = item.dataset.value; // Update global selectedModel
                    modelBtn.innerHTML = `${escapeHtml(item.textContent)} <i class="arrow"></i>`;
                    modelMenu.classList.remove('show');
                    modelBtn.classList.remove('open');
                    handleModelChange(selectedModel);
                });
                modelMenu.appendChild(item);
            }
        }
    }

    modelBtn.addEventListener('click', () => {
      modelMenu.classList.toggle('show');
      modelBtn.classList.toggle('open');
    });
    document.addEventListener('click', e => {
      if (!modelBtn.contains(e.target) && !modelMenu.contains(e.target)) {
        modelMenu.classList.remove('show');
        modelBtn.classList.remove('open');
      }
    });

    userInput.addEventListener('keydown', ev => {
      if (ev.key === 'Enter' && ev.ctrlKey) {
        ev.preventDefault();
        sendMessage();
      }
    });
    sendButton.addEventListener('click', sendMessage);

    // --- Core Logic: sendMessage ---
    async function sendMessage() {
      const messageText = userInput.value.trim();
      if (!messageText || userInput.disabled) {
        if (!selectedModel && !userInput.disabled) displayError('Please select a model first.');
        return;
      }

      const userTimestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const userMessageBubble = createMessageBubble(messageText, 'user');
      addTimestamp(userMessageBubble, userTimestamp);
      
      userInput.value = ''; 
      userInput.style.height = 'auto'; 

      const botThinkingBubble = createMessageBubble('Thinking…', 'bot', true);
      setChatInputEnabled(false);
      const requestStartTime = Date.now();
      
      let accumulatedBotText = "";
      let sourcesData = [];

      try {
        const response = await fetch('/chat', { 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: messageText,
                model: selectedModel, // Use global selectedModel
                session_id: currentSessionId // Use global currentSessionId
            })
        });

        if (!response.ok) { 
            const errorData = await response.json().catch(() => ({ error: `Server error: ${response.status}` }));
            throw new Error(errorData.error || `Server error: ${response.status}`);
        }

        if (!response.body) {
            throw new Error("Response body is null, streaming not possible.");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let firstChunkReceived = false;

        while (true) {
            const { value, done } = await reader.read();
            if (done) {
                console.log("Stream finished.");
                loadChatSessions(); 
                break; 
            }

            const chunkString = decoder.decode(value, { stream: true });
            const eventLines = chunkString.split('\n\n');

            eventLines.forEach(line => {
                if (line.startsWith('data: ')) {
                    try {
                        const eventDataString = line.substring(5).trim();
                        if (!eventDataString) return; 

                        const jsonData = JSON.parse(eventDataString);
                        
                        if (jsonData.type === 'content') {
                            if (!firstChunkReceived) {
                                botThinkingBubble.innerHTML = ''; 
                                botThinkingBubble.classList.remove('thinking');
                                firstChunkReceived = true;
                            }
                            accumulatedBotText += jsonData.text;
                            botThinkingBubble.innerHTML = escapeHtml(accumulatedBotText).replace(/\n/g, '<br>');
                            scrollToBottom();

                        } else if (jsonData.type === 'final') {
                            sourcesData = jsonData.sources || [];
                        } else if (jsonData.type === 'error') {
                            console.error("Error message from stream:", jsonData.message);
                            displayError(jsonData.message);
                        }
                    } catch (e) {
                        console.error("Error parsing SSE event JSON:", e, "Raw line:", line);
                    }
                }
            });
        } 

        const durationSeconds = ((Date.now() - requestStartTime) / 1000).toFixed(1);
        const botTimestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        addTimestamp(botThinkingBubble, `${botTimestamp} (${durationSeconds}s)`);
        addSources(botThinkingBubble, sourcesData);

      } catch (error) {
          console.error('Chat request/stream failed:', error);
          if (botThinkingBubble && botThinkingBubble.parentElement) {
              botThinkingBubble.closest('.message-container')?.remove();
          }
          displayError(error.message || 'Failed to get response from server.');
      } finally {
          setChatInputEnabled(true);
          scrollToBottom();
          userInput.focus();
      }
    }

    // --- handleModelChange ---
    function handleModelChange(modelKey) {
      setChatInputEnabled(false); 
      fetch('/switch_model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelKey, session_id: currentSessionId }) // Use global currentSessionId
      })
      .then(response => {
        if (!response.ok) return response.json().then(err => { throw err; }); 
        return response.json();
      })
      .then(data => {
        if (data.status === 'success') {
          console.log(`Successfully switched to model: ${data.loaded_model_key}`);
        } else {
          throw new Error(data.message || 'Unknown issue switching model.');
        }
      })
      .catch(errorData => { 
        console.error('Model switch error:', errorData);
        const errorMessage = errorData.message || 'Failed to switch model.';
        displayError(errorMessage);
        
        const loadedModelFromServer = errorData.loaded_model_key || selectedModel; // Use global selectedModel
        if (loadedModelFromServer !== modelKey) {
            const previousItem = modelMenu.querySelector(`.model-item[data-value="${loadedModelFromServer}"]`);
            if (previousItem) {
                modelMenu.querySelector('.model-item.selected')?.classList.remove('selected');
                previousItem.classList.add('selected');
                selectedModel = loadedModelFromServer; // Update global selectedModel
                modelBtn.innerHTML = `${escapeHtml(previousItem.textContent)} <i class="arrow"></i>`;
            }
        }
      })
      .finally(() => {
        setChatInputEnabled(true); 
        userInput.focus();
      });
    }

    // --- handleFiles ---
    function handleFiles(files) {
      if (!files.length) return;
      setChatInputEnabled(false);
      fileList.innerHTML = ''; 
      const formData = new FormData();
      let hasValidFiles = false;
      const fileStatusMap = new Map(); 

      Array.from(files).forEach(file => {
        const fileName = file.name;
        const listItem = document.createElement('li');
        listItem.dataset.filename = fileName;
        
        const nameSpan = document.createElement('span');
        nameSpan.textContent = fileName;
        nameSpan.className = 'file-name';
        
        const statusSpan = document.createElement('span');
        statusSpan.className = 'file-status';
        statusSpan.innerHTML = '<span class="status-icon">⏳</span> Pending'; 
        
        listItem.append(nameSpan, statusSpan);
        fileList.appendChild(listItem);
        fileStatusMap.set(fileName, listItem); 

        if (/\.(pdf|txt|docx|md)$/i.test(fileName)) { 
          formData.append('files[]', file);
          hasValidFiles = true;
        } else {
          statusSpan.innerHTML = '<span class="status-icon">❌</span> Invalid Type';
          listItem.style.color = '#dc3545'; 
        }
      });

      if (!hasValidFiles) {
        displayError('No valid files selected (PDF, DOCX, TXT, MD).');
        setChatInputEnabled(true);
        return;
      }

      dropZone.innerHTML = `Uploading ${formData.getAll('files[]').length} valid file(s)...`; 

      fetch('/upload', { method: 'POST', body: formData })
      .then(response => {
        if (!response.ok && response.status !== 207) { 
            return response.json().then(err => { throw new Error(err.error || `Upload failed: ${response.status}`); });
        }
        return response.json();
      })
      .then(data => {
        let overallErrors = false;
        if (data.results && Array.isArray(data.results)) {
          data.results.forEach(result => {
            const listItem = fileStatusMap.get(result.original_filename);
            if (listItem) {
              const statusSpan = listItem.querySelector('.file-status');
              if (result.status === 'success') {
                statusSpan.innerHTML = '<span class="status-icon">📄</span> Ready';
                listItem.style.color = ''; 
              } else {
                statusSpan.innerHTML = `<span class="status-icon">❌</span> Failed`;
                listItem.style.color = '#dc3545';
                overallErrors = true;
              }
            }
          });
        } else if (data.error) { 
            displayError(data.error);
            overallErrors = true;
        } else {
            displayError('Unexpected upload response from server.');
            overallErrors = true;
        }

        if (overallErrors) {
            const errorCount = data.results ? data.results.filter(r => r.status === 'error').length : (data.error ? 1 : 0);
            if (errorCount > 0) displayError(`${errorCount} file(s) failed to process. Check console for details.`);
        }
      })
      .catch(error => {
        console.error('Upload Fetch Error:', error);
        displayError(error.message || 'Network error during upload.');
        fileStatusMap.forEach(listItem => {
          const statusSpan = listItem.querySelector('.file-status');
          if (statusSpan && statusSpan.textContent.includes('Pending')) {
            statusSpan.innerHTML = '<span class="status-icon">❌</span> Network Error';
            listItem.style.color = '#dc3545';
          }
        });
      })
      .finally(() => {
        dropZone.innerHTML = 'Drag & drop Document(s)<br>(PDF, DOCX, TXT, MD) or click'; 
        setChatInputEnabled(true);
        userInput.focus();
        fileInput.value = ''; 
      });
    }

    // --- Inicialización al cargar la página ---
    // Las variables selectedModel, currentSessionId, availableLLMs
    // son ahora globales y se inicializan desde el HTML
    
    populateModelDropdown(); // Populate model dropdown from JS
    
    setChatInputEnabled(true);
    userInput.focus();
    console.log("Chat UI Initialized. Initial Session ID:", currentSessionId);
    
    loadChatSessions(); 

    // Determine if we should load messages for the initial currentSessionId
    // This is a bit tricky. If it's a genuinely new session ID from Flask,
    // there won't be messages. If the user reloaded a page with an existing session,
    // we might want to load them.
    // For now, let's assume a fresh page load is for a new chat or to pick from history.
    // The welcome message for new chats is handled in startNewChat if desired.
    // If `currentSessionId` from Flask actually *is* an old session that should be resumed,
    // you'd call `loadAndDisplayMessagesForSession(currentSessionId);` here.
    // However, the initial Flask route "/" now always gives a *new* UUID.
    // So, on first load, the currentSessionId will not match any existing history.
    // If the user wants to continue an old chat, they must click it from the history list.

    // If you want the very *first* chat to have a default welcome message,
    // you can call it if chatBox is empty after initialization.
    if (chatBox.children.length === 0) {
         createMessageBubble("Welcome! How can I assist you today?", "bot", false, true);
    }

});