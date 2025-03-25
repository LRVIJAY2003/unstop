<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}RAG System{% endblock %}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    {% block extra_css %}{% endblock %}
</head>
<body>
    <header>
        <div class="container">
            <div class="logo">
                <h1>Knowledge Assistant</h1>
            </div>
            <nav>
                <ul>
                    <li><a href="{{ url_for('index') }}">Home</a></li>
                </ul>
            </nav>
        </div>
    </header>

    <main>
        <div class="container">
            {% block content %}{% endblock %}
        </div>
    </main>

    <footer>
        <div class="container">
            <p>&copy; {{ now.year }} Knowledge Assistant</p>
        </div>
    </footer>

    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
    {% block extra_js %}{% endblock %}
</body>
</html>






{% extends 'base.html' %}

{% block title %}Knowledge Assistant{% endblock %}

{% block extra_css %}
<link rel="stylesheet" href="{{ url_for('static', filename='css/chat.css') }}">
{% endblock %}

{% block content %}
<div class="chat-container">
    <div class="sidebar">
        <div class="data-sources">
            <h3>Data Sources</h3>
            <div class="checkbox-group">
                <label class="checkbox-container">
                    <input type="checkbox" id="source-confluence" name="source" value="confluence" checked>
                    <span class="checkmark"></span>
                    Confluence
                </label>
                <label class="checkbox-container">
                    <input type="checkbox" id="source-remedy" name="source" value="remedy" checked>
                    <span class="checkmark"></span>
                    Remedy
                </label>
            </div>
        </div>
        <div class="chat-history">
            <h3>Chat History</h3>
            <ul id="history-list">
                <!-- Chat history will be populated dynamically -->
            </ul>
        </div>
    </div>
    
    <div class="chat-main">
        <div class="chat-messages" id="chat-messages">
            <div class="message system">
                <div class="message-content">
                    <p>Hello! I'm your Knowledge Assistant. I can search our Confluence pages and Remedy tickets to answer your questions. What would you like to know?</p>
                </div>
            </div>
            <!-- Messages will be added dynamically -->
        </div>
        
        <div class="chat-input">
            <form id="query-form">
                <div class="input-group">
                    <input type="text" id="query-input" placeholder="Ask a question..." autocomplete="off">
                    <button type="submit" id="send-button">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="send-icon"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                    </button>
                </div>
            </form>
        </div>
    </div>
    
    <div class="info-panel">
        <div class="panel-header">
            <h3>Information</h3>
            <button id="close-panel">×</button>
        </div>
        <div class="panel-content">
            <div id="source-info">
                <!-- Source information will be displayed here -->
                <p>Select a message to see source information.</p>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}
<script src="{{ url_for('static', filename='js/chat.js') }}"></script>
{% endblock %}







/* Base Styles */
:root {
    --primary-color: #1a73e8;
    --primary-light: #62a3ff;
    --primary-dark: #004baf;
    --secondary-color: #7986cb;
    --background-color: #f3f7ff;
    --surface-color: #ffffff;
    --text-color: #202124;
    --text-secondary: #5f6368;
    --border-color: #e0e0e0;
    --success-color: #34a853;
    --error-color: #ea4335;
    --warning-color: #fbbc05;
    
    --spacing-xs: 4px;
    --spacing-sm: 8px;
    --spacing-md: 16px;
    --spacing-lg: 24px;
    --spacing-xl: 32px;
    
    --border-radius-sm: 4px;
    --border-radius-md: 8px;
    --border-radius-lg: 16px;
    
    --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.12), 0 1px 2px rgba(0, 0, 0, 0.24);
    --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
    --shadow-lg: 0 10px 20px rgba(0, 0, 0, 0.1), 0 3px 6px rgba(0, 0, 0, 0.05);
    
    --font-main: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    --font-size-xs: 12px;
    --font-size-sm: 14px;
    --font-size-md: 16px;
    --font-size-lg: 18px;
    --font-size-xl: 24px;
    
    --transition-fast: 150ms ease;
    --transition-normal: 300ms ease;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

html, body {
    font-family: var(--font-main);
    font-size: var(--font-size-md);
    line-height: 1.5;
    color: var(--text-color);
    background-color: var(--background-color);
    height: 100%;
    width: 100%;
}

body {
    display: flex;
    flex-direction: column;
}

.container {
    width: 100%;
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 var(--spacing-md);
}

/* Header */
header {
    background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
    color: white;
    padding: var(--spacing-md) 0;
    box-shadow: var(--shadow-md);
}

header .container {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo h1 {
    font-size: var(--font-size-xl);
    font-weight: 600;
    margin: 0;
}

nav ul {
    list-style: none;
    display: flex;
}

nav li {
    margin-left: var(--spacing-lg);
}

nav a {
    color: white;
    text-decoration: none;
    font-weight: 500;
    transition: opacity var(--transition-fast);
}

nav a:hover {
    opacity: 0.8;
}

/* Main Content */
main {
    flex: 1;
    padding: var(--spacing-xl) 0;
}

/* Footer */
footer {
    background-color: var(--primary-dark);
    color: white;
    padding: var(--spacing-lg) 0;
    text-align: center;
    font-size: var(--font-size-sm);
}

/* Buttons */
.btn {
    display: inline-block;
    background-color: var(--primary-color);
    color: white;
    padding: var(--spacing-sm) var(--spacing-md);
    border-radius: var(--border-radius-sm);
    font-weight: 500;
    text-decoration: none;
    cursor: pointer;
    border: none;
    transition: background-color var(--transition-fast);
}

.btn:hover {
    background-color: var(--primary-dark);
}

.btn-secondary {
    background-color: var(--secondary-color);
}

.btn-secondary:hover {
    background-color: #5c6bc0;
}

.btn-ghost {
    background-color: transparent;
    color: var(--primary-color);
    border: 1px solid var(--primary-color);
}

.btn-ghost:hover {
    background-color: rgba(26, 115, 232, 0.05);
}

/* Cards */
.card {
    background-color: var(--surface-color);
    border-radius: var(--border-radius-md);
    box-shadow: var(--shadow-sm);
    padding: var(--spacing-lg);
    margin-bottom: var(--spacing-lg);
}

.card-title {
    font-size: var(--font-size-lg);
    font-weight: 600;
    margin-bottom: var(--spacing-md);
}

/* Form elements */
input, textarea, select {
    width: 100%;
    padding: var(--spacing-sm) var(--spacing-md);
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius-sm);
    font-family: var(--font-main);
    font-size: var(--font-size-md);
    transition: border-color var(--transition-fast);
}

input:focus, textarea:focus, select:focus {
    outline: none;
    border-color: var(--primary-color);
}

label {
    display: block;
    margin-bottom: var(--spacing-xs);
    font-weight: 500;
}

/* Responsive */
@media (max-width: 768px) {
    .container {
        padding: 0 var(--spacing-md);
    }
    
    .logo h1 {
        font-size: var(--font-size-lg);
    }
    
    nav li {
        margin-left: var(--spacing-md);
    }
}








/* Chat-specific styles */
.chat-container {
    display: grid;
    grid-template-columns: 250px 1fr 300px;
    grid-gap: var(--spacing-md);
    height: calc(100vh - 180px);
    max-height: 800px;
    min-height: 500px;
}

/* Sidebar */
.sidebar {
    background-color: var(--surface-color);
    border-radius: var(--border-radius-md);
    box-shadow: var(--shadow-sm);
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.data-sources {
    padding: var(--spacing-md);
    border-bottom: 1px solid var(--border-color);
}

.data-sources h3 {
    margin-bottom: var(--spacing-md);
    font-size: var(--font-size-md);
    color: var(--text-secondary);
}

.checkbox-group {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
}

.checkbox-container {
    display: flex;
    align-items: center;
    cursor: pointer;
    user-select: none;
}

.checkbox-container input {
    position: absolute;
    opacity: 0;
    cursor: pointer;
    height: 0;
    width: 0;
}

.checkmark {
    position: relative;
    display: inline-block;
    height: 18px;
    width: 18px;
    margin-right: var(--spacing-sm);
    background-color: #fff;
    border: 2px solid var(--primary-color);
    border-radius: 3px;
}

.checkbox-container:hover input ~ .checkmark {
    background-color: #f0f7ff;
}

.checkbox-container input:checked ~ .checkmark {
    background-color: var(--primary-color);
}

.checkmark:after {
    content: "";
    position: absolute;
    display: none;
}

.checkbox-container input:checked ~ .checkmark:after {
    display: block;
}

.checkbox-container .checkmark:after {
    left: 5px;
    top: 1px;
    width: 4px;
    height: 10px;
    border: solid white;
    border-width: 0 2px 2px 0;
    transform: rotate(45deg);
}

.chat-history {
    flex: 1;
    padding: var(--spacing-md);
    overflow-y: auto;
}

.chat-history h3 {
    margin-bottom: var(--spacing-md);
    font-size: var(--font-size-md);
    color: var(--text-secondary);
}

#history-list {
    list-style: none;
}

#history-list li {
    padding: var(--spacing-sm) 0;
    border-bottom: 1px solid var(--border-color);
    font-size: var(--font-size-sm);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    cursor: pointer;
    transition: background-color var(--transition-fast);
}

#history-list li:hover {
    background-color: #f0f7ff;
}

/* Main Chat */
.chat-main {
    display: flex;
    flex-direction: column;
    background-color: var(--surface-color);
    border-radius: var(--border-radius-md);
    box-shadow: var(--shadow-sm);
    overflow: hidden;
}

.chat-messages {
    flex: 1;
    padding: var(--spacing-md);
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: var(--spacing-md);
}

.message {
    display: flex;
    align-items: flex-start;
    max-width: 80%;
}

.message.user {
    align-self: flex-end;
}

.message .avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    margin-right: var(--spacing-sm);
    background-color: var(--primary-light);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
}

.message.user .avatar {
    background-color: var(--secondary-color);
}

.message .message-content {
    background-color: #f0f7ff;
    padding: var(--spacing-md);
    border-radius: var(--border-radius-md);
    position: relative;
}

.message.user .message-content {
    background-color: var(--primary-color);
    color: white;
}

.message.system .message-content {
    background-color: #f5f5f5;
    color: var(--text-secondary);
}

.message .message-content:after {
    content: '';
    position: absolute;
    left: -8px;
    top: 10px;
    border-top: 8px solid transparent;
    border-bottom: 8px solid transparent;
    border-right: 8px solid #f0f7ff;
}

.message.user .message-content:after {
    left: auto;
    right: -8px;
    border-right: none;
    border-left: 8px solid var(--primary-color);
}

.message .message-content p {
    margin-bottom: var(--spacing-sm);
}

.message .message-content p:last-child {
    margin-bottom: 0;
}

.message .message-content ul,
.message .message-content ol {
    margin-left: var(--spacing-lg);
    margin-bottom: var(--spacing-sm);
}

.message .message-content code {
    font-family: monospace;
    background-color: rgba(0, 0, 0, 0.05);
    padding: 2px 4px;
    border-radius: 3px;
}

.message.user .message-content code {
    background-color: rgba(255, 255, 255, 0.2);
}

.message .message-content pre {
    font-family: monospace;
    background-color: rgba(0, 0, 0, 0.05);
    padding: var(--spacing-sm);
    border-radius: var(--border-radius-sm);
    overflow-x: auto;
    margin-bottom: var(--spacing-sm);
}

.message.user .message-content pre {
    background-color: rgba(255, 255, 255, 0.2);
}

.message .message-meta {
    font-size: var(--font-size-xs);
    color: var(--text-secondary);
    margin-top: var(--spacing-xs);
}

.message.user .message-meta {
    color: rgba(255, 255, 255, 0.8);
}

.chat-input {
    padding: var(--spacing-md);
    border-top: 1px solid var(--border-color);
}

.chat-input .input-group {
    display: flex;
    align-items: center;
}

.chat-input input {
    flex: 1;
    padding: var(--spacing-md);
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius-lg);
    transition: border-color var(--transition-fast), box-shadow var(--transition-fast);
}

.chat-input input:focus {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 2px rgba(26, 115, 232, 0.2);
}

.chat-input button {
    background-color: var(--primary-color);
    color: white;
    border: none;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-left: var(--spacing-sm);
    cursor: pointer;
    transition: background-color var(--transition-fast);
}

.chat-input button:hover {
    background-color: var(--primary-dark);
}

.chat-input .send-icon {
    width: 20px;
    height: 20px;
}

/* Info Panel */
.info-panel {
    background-color: var(--surface-color);
    border-radius: var(--border-radius-md);
    box-shadow: var(--shadow-sm);
    overflow: hidden;
    display: flex;
    flex-direction: column;
}

.panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: var(--spacing-md);
    border-bottom: 1px solid var(--border-color);
}

.panel-header h3 {
    font-size: var(--font-size-md);
    color: var(--text-secondary);
    margin: 0;
}

.panel-header button {
    background: none;
    border: none;
    color: var(--text-secondary);
    font-size: 24px;
    cursor: pointer;
    line-height: 1;
}

.panel-content {
    flex: 1;
    padding: var(--spacing-md);
    overflow-y: auto;
}

.source-item {
    padding: var(--spacing-sm);
    margin-bottom: var(--spacing-sm);
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius-sm);
    background-color: #fbfbfb;
}

.source-item h4 {
    font-size: var(--font-size-sm);
    margin-bottom: var(--spacing-xs);
    color: var(--primary-color);
}

.source-item p {
    font-size: var(--font-size-xs);
    color: var(--text-secondary);
}

.source-item .source-link {
    display: block;
    margin-top: var(--spacing-xs);
    font-size: var(--font-size-xs);
    color: var(--primary-color);
    text-decoration: none;
}

.source-item .source-link:hover {
    text-decoration: underline;
}

/* Loading indicator */
.loading {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: var(--spacing-md);
}

.loading-dots {
    display: flex;
}

.loading-dots span {
    width: 8px;
    height: 8px;
    margin: 0 4px;
    background-color: var(--text-secondary);
    border-radius: 50%;
    animation: dot-pulse 1.5s infinite ease-in-out;
}

.loading-dots span:nth-child(2) {
    animation-delay: 0.2s;
}

.loading-dots span:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes dot-pulse {
    0%, 60%, 100% {
        transform: scale(1);
        opacity: 0.6;
    }
    30% {
        transform: scale(1.5);
        opacity: 1;
    }
}

/* Responsive */
@media (max-width: 1024px) {
    .chat-container {
        grid-template-columns: 200px 1fr;
    }
    
    .info-panel {
        display: none;
    }
}

@media (max-width: 768px) {
    .chat-container {
        grid-template-columns: 1fr;
    }
    
    .sidebar {
        display: none;
    }
}









/**
 * Main JavaScript for the application
 */

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Set current year in footer
    const yearElement = document.querySelector('.year');
    if (yearElement) {
        yearElement.textContent = new Date().getFullYear();
    }
    
    // Initialize tooltips
    const tooltips = document.querySelectorAll('[data-tooltip]');
    tooltips.forEach(tooltip => {
        tooltip.addEventListener('mouseenter', function() {
            const tooltipText = this.getAttribute('data-tooltip');
            const tooltipElement = document.createElement('div');
            tooltipElement.classList.add('tooltip');
            tooltipElement.textContent = tooltipText;
            
            document.body.appendChild(tooltipElement);
            
            const rect = this.getBoundingClientRect();
            tooltipElement.style.left = `${rect.left + (rect.width / 2) - (tooltipElement.offsetWidth / 2)}px`;
            tooltipElement.style.top = `${rect.top - tooltipElement.offsetHeight - 8}px`;
            
            tooltipElement.classList.add('active');
            
            this.addEventListener('mouseleave', function() {
                tooltipElement.remove();
            }, { once: true });
        });
    });
    
    // Check for errors in query parameters
    const urlParams = new URLSearchParams(window.location.search);
    const error = urlParams.get('error');
    if (error) {
        const errorElement = document.createElement('div');
        errorElement.classList.add('alert', 'alert-error');
        errorElement.textContent = decodeURIComponent(error);
        
        // Add to document
        const main = document.querySelector('main');
        if (main) {
            main.insertBefore(errorElement, main.firstChild);
        }
        
        // Remove after 5 seconds
        setTimeout(() => {
            errorElement.remove();
        }, 5000);
    }
});









/**
 * Chat functionality
 */

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const chatMessages = document.getElementById('chat-messages');
    const queryForm = document.getElementById('query-form');
    const queryInput = document.getElementById('query-input');
    const sourceConfluence = document.getElementById('source-confluence');
    const sourceRemedy = document.getElementById('source-remedy');
    const historyList = document.getElementById('history-list');
    const sourceInfo = document.getElementById('source-info');
    
    // Chat history
    let chatHistory = [];
    
    // Function to add a message to the chat
    function addMessage(content, type = 'user', metadata = {}) {
        // Create message element
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', type);
        
        // Create message content
        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');
        
        // Set content (support for Markdown)
        if (typeof content === 'string') {
            // Process content as Markdown (simple conversion)
            let processedContent = content;
            
            // Convert Markdown to HTML (basic implementation)
            // Headers
            processedContent = processedContent.replace(/### (.*?)\n/g, '<h3>$1</h3>');
            processedContent = processedContent.replace(/## (.*?)\n/g, '<h2>$1</h2>');
            processedContent = processedContent.replace(/# (.*?)\n/g, '<h1>$1</h1>');
            
            // Bold
            processedContent = processedContent.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            
            // Italic
            processedContent = processedContent.replace(/\*(.*?)\*/g, '<em>$1</em>');
            
            // Lists
            processedContent = processedContent.replace(/^\- (.*?)$/gm, '<li>$1</li>');
            processedContent = processedContent.replace(/(<li>.*?<\/li>)/gs, '<ul>$1</ul>');
            
            // Code blocks
            processedContent = processedContent.replace(/```(.*?)```/gs, '<pre><code>$1</code></pre>');
            
            // Inline code
            processedContent = processedContent.replace(/`(.*?)`/g, '<code>$1</code>');
            
            // Line breaks
            processedContent = processedContent.replace(/\n/g, '<br>');
            
            contentDiv.innerHTML = processedContent;
        } else {
            contentDiv.textContent = 'Error: Invalid message content';
        }
        
        // Add to message
        messageDiv.appendChild(contentDiv);
        
        // Add meta information if available
        if (metadata.timestamp) {
            const metaDiv = document.createElement('div');
            metaDiv.classList.add('message-meta');
            
            const time = new Date(metadata.timestamp);
            metaDiv.textContent = time.toLocaleTimeString();
            
            contentDiv.appendChild(metaDiv);
        }
        
        // Store any source information in data attribute
        if (metadata.sources) {
            messageDiv.dataset.sources = JSON.stringify(metadata.sources);
        }
        
        // Add click event to show source information
        messageDiv.addEventListener('click', function() {
            const sources = this.dataset.sources;
            if (sources) {
                showSourceInfo(JSON.parse(sources));
            }
        });
        
        // Add to chat history if not a system message
        if (type !== 'system') {
            chatHistory.push({
                content,
                type,
                metadata,
                timestamp: new Date().toISOString()
            });
            
            // Update history list
            updateHistoryList();
        }
        
        // Add to chat
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to show loading indicator
    function showLoading() {
        const loadingDiv = document.createElement('div');
        loadingDiv.classList.add('message', 'system', 'loading');
        loadingDiv.id = 'loading-indicator';
        
        const loadingContent = document.createElement('div');
        loadingContent.classList.add('loading-dots');
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            loadingContent.appendChild(dot);
        }
        
        loadingDiv.appendChild(loadingContent);
        chatMessages.appendChild(loadingDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to hide loading indicator
    function hideLoading() {
        const loadingIndicator = document.getElementById('loading-indicator');
        if (loadingIndicator) {
            loadingIndicator.remove();
        }
    }
    
    // Function to update history list
    function updateHistoryList() {
        // Clear current history
        historyList.innerHTML = '';
        
        // Add history items (most recent first)
        chatHistory.filter(item => item.type === 'user').slice().reverse().forEach((item, index) => {
            const listItem = document.createElement('li');
            listItem.textContent = item.content;
            listItem.dataset.index = chatHistory.length - index - 1;
            
            // Add click event to reload query
            listItem.addEventListener('click', function() {
                const itemIndex = parseInt(this.dataset.index);
                const historyItem = chatHistory[itemIndex];
                
                // Set input value
                queryInput.value = historyItem.content;
                
                // Focus input
                queryInput.focus();
            });
            
            historyList.appendChild(listItem);
        });
    }
    
    // Function to show source information
    function showSourceInfo(sources) {
        if (!sources || !sources.length) {
            sourceInfo.innerHTML = '<p>No source information available for this message.</p>';
            return;
        }
        
        // Clear current source info
        sourceInfo.innerHTML = '';
        
        // Add source information
        sources.forEach(source => {
            const sourceDiv = document.createElement('div');
            sourceDiv.classList.add('source-item');
            
            // Add source title
            const sourceTitle = document.createElement('h4');
            sourceTitle.textContent = source.title || 'Untitled Source';
            sourceDiv.appendChild(sourceTitle);
            
            // Add source type
            const sourceType = document.createElement('p');
            sourceType.textContent = `Type: ${source.type || 'Unknown'}`;
            sourceDiv.appendChild(sourceType);
            
            // Add source snippet if available
            if (source.snippet) {
                const sourceSnippet = document.createElement('p');
                sourceSnippet.textContent = source.snippet;
                sourceDiv.appendChild(sourceSnippet);
            }
            
            // Add source link if available
            if (source.url) {
                const sourceLink = document.createElement('a');
                sourceLink.href = source.url;
                sourceLink.target = '_blank';
                sourceLink.classList.add('source-link');
                sourceLink.textContent = 'View Source';
                sourceDiv.appendChild(sourceLink);
            }
            
            sourceInfo.appendChild(sourceDiv);
        });
    }
    
    // Function to submit query
    async function submitQuery(query) {
        try {
            // Get selected data sources
            const sources = [];
            if (sourceConfluence.checked) sources.push('confluence');
            if (sourceRemedy.checked) sources.push('remedy');
            
            // Check if at least one source is selected
            if (sources.length === 0) {
                alert('Please select at least one data source.');
                return;
            }
            
            // Show loading indicator
            showLoading();
            
            // Send query to API
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query,
                    sources
                })
            });
            
            // Hide loading indicator
            hideLoading();
            
            // Check if request was successful
            if (!response.ok) {
                throw new Error(`Server returned ${response.status} ${response.statusText}`);
            }
            
            // Parse response
            const data = await response.json();
            
            // Check if response contains a generated response
            if (data.generated_response) {
                // Prepare source information
                const sources = [];
                
                // Add Confluence sources if available
                if (data.confluence && data.confluence.raw_results) {
                    data.confluence.raw_results.forEach(result => {
                        sources.push({
                            title: result.title,
                            type: 'Confluence',
                            url: result.url,
                            snippet: result.text_content ? result.text_content.substring(0, 100) + '...' : null
                        });
                    });
                }
                
                // Add Remedy sources if available
                if (data.remedy && data.remedy.raw_results) {
                    data.remedy.raw_results.forEach(result => {
                        sources.push({
                            title: `Ticket #${result.id}`,
                            type: 'Remedy',
                            snippet: result.description ? result.description.substring(0, 100) + '...' : null
                        });
                    });
                }
                
                // Add assistant message with generated response
                addMessage(data.generated_response, 'assistant', { sources });
            } else {
                // Add error message
                addMessage('Error: No response generated.', 'system');
            }
        } catch (error) {
            // Hide loading indicator
            hideLoading();
            
            // Add error message
            addMessage(`Error: ${error.message}`, 'system');
            
            console.error('Error submitting query:', error);
        }
    }
    
    // Handle form submission
    queryForm.addEventListener('submit', function(event) {
        event.preventDefault();
        
        // Get query
        const query = queryInput.value.trim();
        
        // Check if query is empty
        if (!query) {
            return;
        }
        
        // Add user message
        addMessage(query, 'user');
        
        // Clear input
        queryInput.value = '';
        
        // Submit query
        submitQuery(query);
    });
    
    // Focus input on page load
    queryInput.focus();
});









