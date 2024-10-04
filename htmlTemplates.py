css = '''
<style>
body {
    background-color: #f0f2f6;
    font-family: 'Roboto', sans-serif;
}
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
    width: 15%;
}
.chat-message .avatar img {
    max-width: 60px;
    max-height: 60px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid #fff;
}
.chat-message .message {
    width: 85%;
    padding: 0 1.5rem;
    color: #fff;
    font-size: 1.1em;
    line-height: 1.5;
}
.stTextInput > div > div > input {
    background-color: #f0f2f6;
    color: #2b313e;
    border: 1px solid #2b313e;
}
.stButton > button {
    background-color: #2b313e;
    color: #fff;
    border-radius: 20px;
    padding: 0.5rem 1rem;
    font-weight: bold;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background-color: #475063;
    color: #fff;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
