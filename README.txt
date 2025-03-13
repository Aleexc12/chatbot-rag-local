MINIMAL FLASK CHATBOT

![Image](https://github.com/user-attachments/assets/48bc04f7-6a69-428a-a67f-bc91f2e4cbc4)

This is a simple chatbot built using Flask and Hugging Face Transformers, designed for a minimal and clean user experience. The chatbot responds to user messages and can be easily deployed locally.

FEATURES

- Simple UI with a chat interface
- Uses Hugging Face Transformers for generating responses
- Built with Flask as the backend framework
- Minimalistic design for easy interaction
- Lightweight and easy to run locally

INSTALLATION

1. CLONE THE REPOSITORY
   ```
   git clone https://github.com/aleexc12/minimal-flask-chatbot.git
   cd minimal-flask-chatbot
   ```

2. CREATE A VIRTUAL ENVIRONMENT (OPTIONAL, BUT RECOMMENDED)
   ```
   python -m venv venv
   source venv/bin/activate   # For Linux/macOS
   venv\Scripts\activate      # For Windows
   ```

3. INSTALL DEPENDENCIES
   ```
   pip install -r requirements.txt
   ```

USAGE

1. RUN THE CHATBOT
   ```
   python server.py
   ```
   The server will start, and you can access the chatbot in your browser at:
   ```
   http://127.0.0.1:5000
   ```

PROJECT STRUCTURE

```
/minimal-flask-chatbot
│── imgs/
│   └── image.png         # Screenshot of the chatbot UI
│── templates/
│   └── index.html        # Frontend (HTML UI)
│── server.py             # Main Flask backend
│── requirements.txt      # Dependencies
│── README.txt            # Project documentation
```

CONTRIBUTING

Feel free to fork this repository and improve the project. Contributions are welcome!

LICENSE

This project is licensed under the MIT License.
