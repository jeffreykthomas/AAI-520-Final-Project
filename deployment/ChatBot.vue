<template>
  <div class="chat full-width q-px-md q-px-xl-md">
    <div class="full-width row wrap justify-center q-py-xs">
      <div class="col-4 col-md-5 text-center q-pt-md">
        <q-img
          src="../assets/images/chat.jpg"
          class="chat-logo"
          alt="Chat Image"
          :ratio="1"
        />
      </div>
    </div>
    <div class="full-width row wrap justify-center q-py-xs">
      <div class="col-12 col-md-8 text-center">
        <h1 class="text-h4 text-weight-bold">Welcome to the UbuntuChat</h1>
        <p class="q-mb-none">All models fine-tuned on the <a href="https://github.com/rkadlec/ubuntu-ranking-dataset-creator" class="text-primary">Ubuntu Dataset</a></p>
        <a class="q-mb-none text-primary"
           href="https://github.com/jeffreykthomas/AAI-520-Final-Project/">
          Check out our github repository</a>
        <p class="text-italic">Shyam Adhikari, Matthew Sexton, Jeffrey Thomas</p>
      </div>
    </div>
    <div class="full-width row wrap justify-center q-py-xs">
      <div class="col-12 col-md-8 text-center">
        <q-btn-group size="lg">
          <q-btn
            :color="dialoMedium ? 'primary' : 'light'"
            class="text-center q-py-sm model-selection"
            :class="dialoMedium ? 'text-white' : 'text-black'"
            @click="switchModel(0)"
            ><span class="text-capitalize">GPT-2 Medium</span></q-btn
          >
          <q-btn
            :color="dialoLarge ? 'primary' : 'light'"
            class="text-center q-btn-border q-py-sm model-selection"
            :class="dialoLarge ? 'text-white' : 'text-black'"
            @click="switchModel(1)"
          ><span class="text-capitalize">GPT-2 Large</span></q-btn
          >
          <q-btn
            :color="gpt ? 'primary' : 'light'"
            class="text-center text-capitalize q-py-sm model-selection"
            :class="gpt ? 'text-white' : 'text-black'"
            @click="switchModel(2)"
            >GPT-3.5 Turbo</q-btn
          >
          <q-btn
            :color="llama2 ? 'primary' : 'light'"
            class="text-center text-capitalize q-py-sm model-selection"
            :class="llama2 ? 'text-white' : 'text-black'"
            @click="switchModel(3)"
          >Llama2 7B</q-btn
          >
        </q-btn-group>
      </div>
    </div>
    <div class="full-width row wrap justify-center q-pt-lg q-pb-sm">
      <div class="col-12 col-md-8 chat-container"
           :class="[ showIndicator ? 'q-pb-0' : 'q-pb-sm' ]"
      >

        <div v-for="(message, i) in currentMessageList" :key="i">
          <transition
            appear
            :enter-active-class="message.user === 'Anon' ? '' : 'animated fadeIn animate__slower'"
            :leave-active-class="message.user === 'Anon' ? '' : 'animated fadeOut animate__slower'"
          >
            <q-chat-message
              class="message-container q-pb-md"
              :name="message.user"
              :sent="message.user === 'Anon'"
              :bg-color="message.user === 'Anon' ? 'primary' : 'grey-4'"
              :text-color="message.user === 'Anon' ? 'white' : 'black'"
            >
              <div v-html="formattedText(message.text)"></div>
            </q-chat-message>
          </transition>
        </div>
      </div>
    </div>
    <div class="row q-px-md-xl q-pb-xl q-mb-xl justify-center">
      <!-- Some example questions to try-->
      <div class="col-12 col-md-8">
        <div v-show="showIndicator" class="q-pb-xs">
          <div>
            <div class="typing-indicator q-pb-xl">
              <p v-if="currentMessageList.length < 3 && !gpt">Hugging Face models may take up to 20 seconds to load</p>
              <p v-else>Model response is generating</p>
              <q-spinner grow color="primary" size="sm" class="mx-1" />
              <q-spinner grow color="primary" size="sm" class="me-1" />
              <q-spinner grow color="primary" size="sm" />
            </div>
          </div>
        </div>
      </div>
      <div class="col-12">
        <h3 class="text-center text-h5 q-my-sm text-italic text-weight-bold">Try asking me:</h3>
      </div>
      <div class="col-12 col-md-3 q-px-md-lg q-px-lg-xl q-py-sm">
        <q-btn label="What is the best feature in Ubuntu?" color="positive" size="lg" padding="sm"
               @click="onMessageWasSent('What is the best feature in Ubuntu?')"
               class="text-capitalize full-width rounded-1 fs-1"/>
      </div>
      <div class="col-12 col-md-3 q-px-md-lg q-px-lg-xl q-py-sm">
        <q-btn label="Can I change my desktop image?" color="positive" size="lg" padding="sm"
               @click="onMessageWasSent('How do I change my desktop image?')"
               class="text-capitalize full-width rounded-1 fs-1"/>
      </div>
      <div class="col-12 col-md-3 q-px-md-lg q-px-lg-xl q-py-sm">
        <q-btn label="what kind of modem can I use?" color="positive" size="lg" padding="sm"
               @click="onMessageWasSent('What kind of modem can I use?')"
               class="text-capitalize full-width rounded-1 fs-1"/>
      </div>
    </div>
    <div class="row">
      <div class="col q-py-lg">
        <q-input
          outlined
          rounded
          autogrow
          ref="chatInput"
          v-model="newMessage"
          :label="chatFocused ? 'Ask a question' : '&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Ask a question'"
          @focus="handleFocus"
          @blur="handleBlur"
          @keyup.enter="onMessageWasSent(newMessage)"
          class="sc-user-input"
          :class="{ 'drawer-expanded': store.getLeftDrawer }"
        >
          <template v-slot:prepend>
            <q-icon
              v-if="!chatFocused && newMessage === ''"
              name="fa-sharp fa-light fa-comment-dots"
              class="message-icon"
              size="1.8rem"
            />
          </template>
          <template v-slot:append>
            <q-icon
              v-if="newMessage"
              name="fa-sharp fa-light fa-paper-plane"
              class="cursor-pointer q-mr-sm"
              :class="{ 'drawer-expanded': store.getLeftDrawer }"
              size="1.8rem"
              @click="onMessageWasSent(newMessage)"
            />
          </template>
        </q-input>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, nextTick, ref, Ref, computed } from 'vue';
import { auth, signInAnonymously } from 'boot/firebase';
import { Message } from 'components/models';
import { useJTStore } from 'stores/jt_store';
//variables

const store = useJTStore();
const lastMessage = ref('');
const currentMessageList = ref<Message[]>([]);
const dialoMedium = ref(true);
const dialoLarge = ref(false);
const gpt = ref(false);
const llama2 = ref(false);
const chatFocused = ref(false);
const showIndicator = ref(false);
const newMessage = ref('');
const chatInput: Ref<HTMLInputElement | null> = ref(null)
const newRoom = {
  roomId: '',
  participants: [
    {
      id: 'user1',
      name: 'MentorsChat',
      imageUrl: 'https://avatars3.githubusercontent.com/u/1915989?s=230&v=4',
    },
  ], // the list of the messages to show, can be paginated and adjusted dynamically
};
const streamFinished = ref(true);
const userBot = computed(() => {
  let name = ''
  if (dialoMedium.value) {
    name = 'Dialo Medium'
  } else if (dialoLarge.value) {
    name = 'Dialo Large'
  } else if (gpt.value) {
    name = 'GPT-3.5 Turbo'
  } else {
    name = 'Llama2 7B'
  }
  return name
});

function handleFocus() {
  console.log('focus');
  chatFocused.value = true;
}
function handleBlur() {
  console.log('blur');
  chatFocused.value = false;
}
async function switchModel(i: number) {
  switch (i) {
    case 0:
      dialoMedium.value = true;
      dialoLarge.value = false;
      gpt.value = false;
      break;
    case 1:
      dialoMedium.value = false;
      dialoLarge.value = true;
      gpt.value = false;
      break;
    case 2:
      dialoMedium.value = false;
      dialoLarge.value = false;
      gpt.value = true;
      break;
    case 3:
      dialoMedium.value = false;
      dialoLarge.value = false;
      gpt.value = false;
      llama2.value = true;
      break;
  }
  lastMessage.value = '';
  currentMessageList.value = [];
  await sendTypedMessage('Ask me about Ubuntu')
}
async function onMessageWasSent(message: string) {
  console.log('message sent:' + message);
  const messageData = {
    user: 'Anon',
    text: message,
  };
  newMessage.value = '';
  currentMessageList.value.push(messageData);
  showIndicator.value = true;
  scrollToBottom();
  console.log(message);
  let response;

  let model = 'medium';
  if (dialoLarge.value) {
    model = 'large';
  } else if (gpt.value) {
    model = 'gpt';
  } else if (llama2.value) {
    model = 'llama2';
  }
  // Send the maximum number of messages to the model, limit at 1024 tokens
  response = await fetch(
    'https://us-central1-chat-emts.cloudfunctions.net/ubuntu-chat',
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ conversation: currentMessageList.value, model: model }),
    }
  );
  if (!response.ok) {
    console.log(`An error occurred: ${response.status}`);
    return;
  }

  const data = await response.json();
  showIndicator.value = false;
  // Handle the received data
  console.log('data received from server: ', data);
  let messageModel = gpt.value || llama2.value
  let text = messageModel ? data['message'] : data[0]['generated_text'];

  let botMessage = {
    user: userBot.value,
    text: capitalizeFirstLetter(text),
  };
  currentMessageList.value.push(botMessage);

  // Handle the received data
  // console.log('data received from server: ', data.body)

  scrollToBottom();
}
function capitalizeFirstLetter(stringToCapitalize: string) {
  return (
    stringToCapitalize.charAt(0).toUpperCase() + stringToCapitalize.slice(1)
  );
}

function formattedText(text: string) {
  return text.replace(/\r\n|\r|\n/g, '<br>');
}
async function sendTypedMessage(messageContent: string) {
  streamFinished.value = false;
  let messageList = messageContent.split(' ');
  const delay = (ms: number) =>
    new Promise((resolve) => setTimeout(resolve, ms));

  async function executeSlowly(list: Array<string>) {
    for (const word of list) {
      await delay(50);
      await addToLastMessage(word + ' ');
    }
  }

  await executeSlowly(messageList);
  streamFinished.value = true;
}
// async function onGPTMessageWasSent(message: Message) {
//   const messageData = {
//     user: 'Anon',
//     text: message.text,
//   };
//
//   console.log('user scrolled set to false');
//   scrollToBottom();
//
//   let eventSource = new EventSource(
//     `https://us-central1-chat-emts.cloudfunctions.net/streaming-gpt?question=${encodeURIComponent(
//       JSON.stringify({ text: messageData.text })
//     )}`
//   );
//
//   eventSource.addEventListener('message', onData);
//   eventSource.addEventListener('error', onError);
//   eventSource.addEventListener('open', onOpen);
//
//   function onOpen() {
//     console.log('Connection to server opened');
//   }
//
//   function onData(event: MessageEvent) {
//     // Handle data received from the stream
//     let eventData: string = event.data;
//     showIndicator.value = false;
//     if (eventData === 'STREAM_COMPLETE') {
//       console.log('Stream complete!');
//       onComplete();
//     } else {
//       let newData = eventData;
//       const regex = /\b(?:e\.g\.|i\.e\.)|([.!?:](?!\s))/gm;
//       const charRegex = /[A-Za-z0-9]/;
//       if (
//         (regex.test(lastMessage.value.slice(-1)) && !eventData.startsWith(' ')) ||
//         (/[a-zA-Z]/.test(lastMessage.value.slice(-1)) && eventData.match(/^\d/)) ||
//         (charRegex.test(lastMessage.value.slice(-1)) && eventData.startsWith('- '))
//       ) {
//         newData = '\n\n' + newData;
//       }
//       addToLastMessage(newData);
//       scrollToBottom();
//       streamFinished.value = false;
//     }
//   }
//   let onComplete = () => {
//     // Stream finished, remove event listeners
//     streamFinished.value = true;
//     eventSource.removeEventListener('message', onData);
//     eventSource.removeEventListener('error', onError);
//     eventSource.removeEventListener('open', onOpen);
//     eventSource.close();
//   };
//
//   function onError(error: Event) {
//     setTimeout(() => {
//       if (eventSource.readyState === EventSource.CLOSED) {
//         console.log('Connection was closed by the server');
//         // Handle connection closed by server
//         onComplete();
//       } else {
//         console.log(error);
//         console.log(
//           'An error occurred while streaming the data from the server: ' + error
//         );
//       }
//     }, 2000);
//   }
//
//   console.log('new message sent: ' + messageData);
//   showIndicator.value = true;
// }
async function addToLastMessage(text: string) {
  console.log('adding to last message: ' + text);
  if (streamFinished.value) {
    lastMessage.value = '';
  }
  lastMessage.value = lastMessage.value + text;
  const botMessage = {
    user: userBot.value,
    text: lastMessage.value,
  };
  if (streamFinished.value) {
    currentMessageList.value.push(botMessage);
  } else {
    currentMessageList.value.splice(currentMessageList.value.length - 1, 1, botMessage);
    console.log('current message list: ', currentMessageList);
  }
  await nextTick();
}
function scrollToBottom() {
  // Get the chat container element
  const chatContainer = document.querySelector('.chat-container');
  console.log('chat container: ', chatContainer)
  // If the chat container exists
  if (chatContainer) {
    // Calculate the bottom position of the container relative to the whole document
    const bottomPosition =
      chatContainer.getBoundingClientRect().bottom + window.pageYOffset;

    // Scroll the window to the bottom position
    window.scrollTo({ top: bottomPosition, behavior: 'smooth' });
  }
}
async function openChat() {
  console.log('open chat called');

  let newClientData = {
    id: 'user2',
    name: 'Anon',
    email: '',
    phone: '',
    zip: '',
    clientRef: '',
    imageUrl: '',
  };

  newRoom.participants.push(newClientData);
}

onMounted(async () => {
  await openChat();
  await sendTypedMessage('Ask me about Ubuntu');

  signInAnonymously(auth)
    .then(() => {
      console.log('Signed in as anonymous');
    })
    .catch((error) => {
      const errorCode = error.code;
      const errorMessage = error.message;
      console.log(`Error code: ${errorCode}, message: ${errorMessage}`);
    });
});
</script>
