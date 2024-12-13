// Import Firebase Modules
import { initializeApp } from "https://www.gstatic.com/firebasejs/9.17.1/firebase-app.js";
import {
  getFirestore,
  collection,
  addDoc,
  onSnapshot,
} from "https://www.gstatic.com/firebasejs/9.17.1/firebase-firestore.js";

// Firebase Configuration
const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_AUTH_DOMAIN",
  projectId: "YOUR_PROJECT_ID",
  storageBucket: "YOUR_STORAGE_BUCKET",
  messagingSenderId: "YOUR_MESSAGING_SENDER_ID",
  appId: "YOUR_APP_ID"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

const postThreadBtn = document.getElementById("post-thread");
const threadTitle = document.getElementById("thread-title");
const threadContent = document.getElementById("thread-content");

postThreadBtn.addEventListener("click", async (event) => {
  event.preventDefault();
  console.log("Submit button clicked");

  if (!threadTitle.value || !threadContent.value) {
    alert("Please fill in both title and content.");
    console.log("Missing title or content");
    return;
  }

  try {
    console.log("Adding document to Firestore");
    await addDoc(collection(db, "threads"), {
      title: threadTitle.value,
      content: threadContent.value,
      createdAt: new Date(),
    });
    console.log("Document successfully added");
    threadTitle.value = "";
    threadContent.value = "";
    alert("Thread posted successfully!");
  } catch (error) {
    console.error("Error posting thread:", error);
  }
});

const threadList = document.getElementById("thread-list");

// Real-time Listener
onSnapshot(collection(db, "threads"), (snapshot) => {
  threadList.innerHTML = "";
  snapshot.forEach((doc) => {
    const thread = doc.data();
    const li = document.createElement("li");
    li.classList.add("p-4", "bg-gray-100", "rounded-lg", "shadow");
    li.innerHTML = `
      <h3 class="text-lg font-bold" flex justify-center>${thread.title}</h3>
      <p class="text-gray-700 mt-2">${thread.content}</p>
      <p class="text-sm text-gray-500 mt-2">Posted on: ${thread.createdAt
        .toDate()
        .toLocaleString()}</p>
    `;
    threadList.appendChild(li);
  });
});
