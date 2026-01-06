// API Configuration - Use relative path when served from same origin
const API_BASE_URL = window.location.origin;

// Session Management - Store session ID in localStorage
let currentSessionId = localStorage.getItem('chatbot_session_id') || null;

// DOM Elements
const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadStatus = document.getElementById('uploadStatus');
const docCount = document.getElementById('docCount');
const refreshStats = document.getElementById('refreshStats');
const topKInput = document.getElementById('topK');
const clearSessionBtn = document.getElementById('clearSession');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    setupEventListeners();
    // Initialize or restore session
    if (!currentSessionId) {
        initializeSession();
    }
});

// Event Listeners
function setupEventListeners() {
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    fileInput.addEventListener('change', handleFileUpload);
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    refreshStats.addEventListener('click', loadStats);
    clearSessionBtn.addEventListener('click', handleClearSession);
}

// Session Management Functions
async function initializeSession() {
    try {
        const response = await fetch(`${API_BASE_URL}/sessions`, {
            method: 'POST',
        });
        
        if (response.ok) {
            const data = await response.json();
            currentSessionId = data.session_id;
            localStorage.setItem('chatbot_session_id', currentSessionId);
            console.log('Session initialized:', currentSessionId);
        }
    } catch (error) {
        console.error('Error initializing session:', error);
    }
}

function clearSession() {
    currentSessionId = null;
    localStorage.removeItem('chatbot_session_id');
    initializeSession();
}

// Chat Functions
async function sendMessage() {
    const question = userInput.value.trim();
    if (!question) return;

    // Ensure we have a session
    if (!currentSessionId) {
        await initializeSession();
    }

    // Add user message to chat
    addMessage(question, 'user');
    userInput.value = '';
    sendButton.disabled = true;

    // Show loading indicator
    const loadingId = addLoadingMessage();

    try {
        const topK = parseInt(topKInput.value) || 5;
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                top_k: topK,
                session_id: currentSessionId,
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        // Update session ID if returned (should be same, but just in case)
        if (data.session_id) {
            currentSessionId = data.session_id;
            localStorage.setItem('chatbot_session_id', currentSessionId);
        }
        
        // Remove loading message
        removeMessage(loadingId);
        
        // Add bot response
        addMessage(data.answer, 'bot', data.source_documents);
        
    } catch (error) {
        console.error('Error:', error);
        removeMessage(loadingId);
        addMessage('Sorry, I encountered an error. Please try again.', 'bot');
    } finally {
        sendButton.disabled = false;
        userInput.focus();
    }
}

function addMessage(content, type, sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const p = document.createElement('p');
    p.textContent = content;
    contentDiv.appendChild(p);
    
    // Add source documents if available
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'source-documents';
        
        const h4 = document.createElement('h4');
        h4.textContent = '📚 Sources:';
        sourcesDiv.appendChild(h4);
        
        sources.forEach((source, index) => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';
            const fileName = source.metadata?.source_file?.split('/').pop() || 'Unknown';
            sourceItem.textContent = `${index + 1}. ${fileName}`;
            sourcesDiv.appendChild(sourceItem);
        });
        
        contentDiv.appendChild(sourcesDiv);
    }
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageDiv;
}

function addLoadingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    messageDiv.id = 'loading-message';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading';
    contentDiv.appendChild(loadingDiv);
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return 'loading-message';
}

function removeMessage(id) {
    const message = document.getElementById(id);
    if (message) {
        message.remove();
    }
}

// File Upload Functions
async function handleFileUpload(event) {
    const files = event.target.files;
    if (files.length === 0) return;
    
    await uploadFiles(Array.from(files));
}

function handleDragOver(event) {
    event.preventDefault();
    uploadArea.style.borderColor = '#4f46e5';
    uploadArea.style.background = 'rgba(79, 70, 229, 0.1)';
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.style.borderColor = '';
    uploadArea.style.background = '';
}

async function handleDrop(event) {
    event.preventDefault();
    uploadArea.style.borderColor = '';
    uploadArea.style.background = '';
    
    const files = Array.from(event.dataTransfer.files);
    const allowedTypes = ['.pdf', '.txt', '.docx', '.md'];
    const validFiles = files.filter(file => {
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        return allowedTypes.includes(ext);
    });
    
    if (validFiles.length > 0) {
        await uploadFiles(validFiles);
    } else {
        showUploadStatus('Please upload PDF, TXT, DOCX, or MD files only.', 'error');
    }
}

async function uploadFiles(files) {
    showUploadStatus('Uploading files...', 'success');
    
    for (const file of files) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch(`${API_BASE_URL}/upload`, {
                method: 'POST',
                body: formData,
            });
            
            if (!response.ok) {
                throw new Error(`Failed to upload ${file.name}`);
            }
            
            const data = await response.json();
            showUploadStatus(
                `✓ ${file.name} uploaded successfully! (${data.chunks_added} chunks)`,
                'success'
            );
            
            // Refresh stats
            setTimeout(loadStats, 1000);
            
        } catch (error) {
            console.error('Upload error:', error);
            showUploadStatus(`✗ Failed to upload ${file.name}`, 'error');
        }
    }
    
    // Clear file input
    fileInput.value = '';
}

function showUploadStatus(message, type) {
    uploadStatus.textContent = message;
    uploadStatus.className = `upload-status ${type}`;
    
    if (type === 'success') {
        setTimeout(() => {
            uploadStatus.style.display = 'none';
        }, 5000);
    }
}

// Session Functions
async function handleClearSession() {
    if (confirm('Are you sure you want to clear the conversation? This will start a new session.')) {
        if (currentSessionId) {
            try {
                await fetch(`${API_BASE_URL}/sessions/${currentSessionId}`, {
                    method: 'DELETE',
                });
            } catch (error) {
                console.error('Error deleting session:', error);
            }
        }
        
        // Clear chat messages (keep welcome message)
        const messages = chatMessages.querySelectorAll('.message');
        messages.forEach((msg, index) => {
            if (index > 0) { // Keep first welcome message
                msg.remove();
            }
        });
        
        // Start new session
        clearSession();
    }
}

// Stats Functions
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/vectorstore/stats`);
        if (!response.ok) {
            throw new Error('Failed to load stats');
        }
        
        const data = await response.json();
        docCount.textContent = data.total_documents || 0;
        
    } catch (error) {
        console.error('Error loading stats:', error);
        docCount.textContent = '-';
    }
}

