document.addEventListener('DOMContentLoaded', function() {
  document.getElementById('verify').addEventListener('click', function() {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      chrome.tabs.sendMessage(tabs[0].id, {action: "verify"});
      document.getElementById('loading').style.display = 'block';
    });
  });
});

chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.action === "verificationComplete") {
    document.getElementById('loading').style.display = 'none';
  }
});