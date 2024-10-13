chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.action === "verify") {
    let articles = document.querySelectorAll('.featured-article, .articles article');
    articles.forEach((article, index) => {
      let content = article.querySelector('p').textContent;
      let heading = article.querySelector('h2, h3');
      
      fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({text: content}),
      })
      .then(response => response.json())
      .then(data => {
        let icon = document.createElement('span');
        icon.style.marginLeft = '10px';
        icon.innerHTML = data.label === 'Real' ? '✅' : '❌';
        heading.appendChild(icon);
      });
    });

    // Notify that verification is complete
    chrome.runtime.sendMessage({action: "verificationComplete"});
  }
});