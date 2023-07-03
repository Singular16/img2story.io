from dotenv import find_dotenv, load_dotenv
import pipelines
import requests, os, streamlit as st
from langchain import PromptTemplate, LLMChain, HuggingFaceHub, OpenAI


load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv()

# image to text
def img2text(url:str):
	img_to_txt = pipeline(
		"image-to-text",
		model = "Salesforce/blip-image-captioning-base"
	)

	text = img_to_txt(url)[0]['generated_text']

	print(text)
	return text


# img2text("photo.png")
# img src can be a single file or a link

#llm

def generate_story(scenario:str):
	template = """
	You are a story teller:
	you can generate a short story based on a single narrative,
	the story should be no more than 20 words

	CONTEXT: {scenario}
	STORY: 
	"""

	prompt = PromptTemplate(
		template = template,
		input_variables= ["scenario"])

	# use  openai
	# story_llm = LLMChain(
	# 	llm = OpenAI(
	# 		model_name = 'gpt-3.5-turbo',
	# 		temperature = 1
	# 	),
	# 	prompt = prompt,
	# 	verbose = True
	# )

	repo_id:str = "tituae/falcon-7b-instruct"
	falcon_llm = HuggingFaceHub(
		repo_id=repo_id,
		model_kwargs={
			"temperature":0.1,
			"max_new_tokens":500
		}
	)
		
	story_llm = LLMChain(
		prompt=prompt,
		llm=falcon_llm,
		verbose = True
	)


	story = story_llm.predict(scenario = scenario)

	print(story)
	return story


# text2speech
def text2speech(message:str):
	API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
	headers = {"Authorization":f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

	payloads = {"inputs": message}
	response = requests.post(
		API_URL,
		headers=headers,
		json = payloads
	)

	with open("audio.flac", 'wb') as file:
		file.write(response.content)
	

# scenario = img2text("photo.png")
# story = generate_story(scenario)
# text2speech(story)


def main():
	st.set_page_config(
		page_title="img2audio-story"
	)
	st.header("Turn image into audio story")
	uploaded_file = st.file_uploader("choose an image", type="jpg")

	if uploaded_file is not None:
		print(uploaded_file)
		bytes_data = uploaded_file.getvalue()

		with open(uploaded_file.name, 'wb') as file:
			file.write(bytes_data)
		st.image(
			uploaded_file,
			caption = "Uploaded Image.",
			use_column_width=True
		)
		scenario = img2text(uploaded_file.name)
		story = generate_story(scenario)
		text2speech(story)

		with st.expander("scenario"):
			st.write(scenario)
		with st.expander("story"):
			st.write(story)

		st.audio("audio.flac")
	

if __name__ == "__main__":
	main()
