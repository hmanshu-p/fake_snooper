chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.action === "verificationComplete") {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      chrome.tabs.reload(tabs[0].id);
    });
  }
});