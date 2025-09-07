document.addEventListener("DOMContentLoaded", () => {
  const chatForm = document.getElementById("chatForm");
  const messageInput = document.getElementById("messageInput");
  const chatContainer = document.getElementById("chatContainer");
  const recentCries = document.getElementById("recentCries");

  // Quick question buttons
  document.querySelectorAll('.quick-question').forEach(btn => {
    btn.addEventListener('click', () => {
      const question = btn.dataset.question;
      messageInput.value = question;
      sendMessage(question);
    });
  });

  // Chat form submission
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const message = messageInput.value.trim();
    if (!message) return;

    sendMessage(message);
    messageInput.value = '';
  });

  function sendMessage(message) {
    // Add user message to chat
    addMessage(message, 'user');
    
    // Show typing indicator
    const typingId = showTypingIndicator();
    
    // Simulate AI response (replace with actual Gemini API call)
    setTimeout(() => {
      removeTypingIndicator(typingId);
      const response = generateAIResponse(message);
      addMessage(response, 'ai');
    }, 1500);
  }

  function addMessage(content, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender}-message mb-3`;
    
    const isAI = sender === 'ai';
    const icon = isAI ? 'bi-robot' : 'bi-person';
    const bgColor = isAI ? 'bg-primary' : 'bg-secondary';
    const messageBg = isAI ? 'bg-light' : 'bg-primary text-white';
    
    messageDiv.innerHTML = `
      <div class="d-flex align-items-start">
        <div class="avatar ${bgColor} text-white rounded-circle p-2 me-2">
          <i class="bi ${icon}"></i>
        </div>
        <div class="message-content">
          <div class="message-bubble ${messageBg} p-3 rounded">
            <p class="mb-0">${content}</p>
          </div>
          <small class="text-muted">${new Date().toLocaleTimeString()}</small>
        </div>
      </div>
    `;
    
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    const typingId = 'typing-' + Date.now();
    typingDiv.id = typingId;
    typingDiv.className = 'chat-message ai-message mb-3';
    typingDiv.innerHTML = `
      <div class="d-flex align-items-start">
        <div class="avatar bg-primary text-white rounded-circle p-2 me-2">
          <i class="bi bi-robot"></i>
        </div>
        <div class="message-content">
          <div class="message-bubble bg-light p-3 rounded">
            <div class="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        </div>
      </div>
    `;
    
    chatContainer.appendChild(typingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return typingId;
  }

  function removeTypingIndicator(typingId) {
    const typingDiv = document.getElementById(typingId);
    if (typingDiv) {
      typingDiv.remove();
    }
  }

  function generateAIResponse(message) {
    const lowerMessage = message.toLowerCase();
    
    // Baby cry related responses
    if (lowerMessage.includes('cry') || lowerMessage.includes('crying')) {
      return `Based on your question about crying, here are common reasons babies cry:
      
• <strong>Hunger:</strong> Most common cause - check feeding schedule
• <strong>Discomfort:</strong> Wet diaper, tight clothing, or temperature
• <strong>Sleep:</strong> Overtired or needs help falling asleep
• <strong>Gas/Pain:</strong> Try gentle tummy massage or burping
• <strong>Overstimulation:</strong> Too much noise, light, or activity

Try our cry analysis feature to get specific insights about your baby's cries!`;
    }
    
    // Temperature related responses
    if (lowerMessage.includes('temperature') || lowerMessage.includes('fever')) {
      return `Normal baby temperature ranges:
      
• <strong>Rectal:</strong> 97.9°F - 100.4°F (36.6°C - 38°C)
• <strong>Oral:</strong> 95.9°F - 99.5°F (35.5°C - 37.5°C)
• <strong>Armpit:</strong> 94.5°F - 99.1°F (34.7°C - 37.3°C)

<strong>Call doctor if:</strong>
• Under 3 months: Any fever above 100.4°F
• 3-6 months: Fever above 101°F
• Any age: Fever lasting more than 5 days`;
    }
    
    // Feeding related responses
    if (lowerMessage.includes('feed') || lowerMessage.includes('hungry')) {
      return `General feeding guidelines:
      
• <strong>Newborns:</strong> Every 2-3 hours (8-12 times/day)
• <strong>1-2 months:</strong> Every 3-4 hours (6-8 times/day)
• <strong>3-6 months:</strong> Every 4-5 hours (5-6 times/day)
• <strong>6+ months:</strong> Every 4-6 hours with solid foods

<strong>Signs of hunger:</strong>
• Rooting reflex, sucking on hands
• Crying that stops when fed
• Increased alertness`;
    }
    
    // Emergency related responses
    if (lowerMessage.includes('doctor') || lowerMessage.includes('emergency')) {
      return `🚨 <strong>Call 911 or go to ER immediately if:</strong>
      
• Difficulty breathing or blue lips
• Unconsciousness or severe lethargy
• Seizures
• Severe bleeding
• Head injury with vomiting

<strong>Call doctor within 24 hours:</strong>
• Fever in babies under 3 months
• Persistent vomiting or diarrhea
• Refusing to eat for 8+ hours
• Unusual rash or behavior changes`;
    }
    
    // Default response
    return `Thank you for your question about baby care! I'm here to help with:
    
• Understanding your baby's needs
• Health and safety guidance
• Feeding and sleep advice
• Development milestones
• Emergency situations

For specific medical advice, always consult with your pediatrician. You can also use our cry analysis feature to better understand your baby's communication patterns.`;
  }

  // Load recent cry analysis data
  function loadRecentCries() {
    // This would typically fetch from your backend
    // For now, showing sample data
    const sampleCries = [
      { type: 'hungry', time: '2:30 PM', confidence: '95%' },
      { type: 'tired', time: '1:15 PM', confidence: '87%' },
      { type: 'discomfort', time: '12:45 PM', confidence: '92%' }
    ];
    
    if (sampleCries.length > 0) {
      recentCries.innerHTML = sampleCries.map(cry => `
        <div class="d-flex justify-content-between align-items-center mb-2">
          <span class="badge bg-primary">${cry.type}</span>
          <small class="text-muted">${cry.time}</small>
        </div>
        <small class="text-muted">Confidence: ${cry.confidence}</small>
      `).join('');
    }
  }

  // Initialize
  loadRecentCries();
});
