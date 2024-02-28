# Streamlit Chat Application

This is a Streamlit-based chat application that utilizes OpenAI's GPT-3 language model for generating responses to user queries and have real time feedback on each response.

## Setup

To run this project locally, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies by running:
    ```
    pip install -r requirements.txt
    ```
3. Set up your OpenAI API key:
    ```
    os.environ["OPENAI_API_KEY"]="your-api-key"
    ```
   or directly modifying the code in `main.py` `ref_app.py` to assign your API key to the `os.environ["OPENAI_API_KEY"]` variable.
4. Run the Streamlit app by executing:
    ```
    streamlit run main.py
    ```
5. Run the reference version:
    ```
    streamlit run ref_app.py
    ```

## Configuration

- To add a custom system propt, modify the `system_prompt` variable in the `main.py` file.
- The OpenAI API key is set using the `os.environ` variable.
- The application allows switching between feedback modes ('Thumbs' or 'Faces') using a toggle button.
- A CSV file is used for storing user queries, generated responses, feedback, and timestamps.

## Usage

- Upon running the application, users can input queries in the chat interface.
- The chatbot will respond with generated answers.
- Users can provide feedback on the responses using the provided feedback options.
- Feedback is stored in the CSV file along with the corresponding query, response, and timestamp.

