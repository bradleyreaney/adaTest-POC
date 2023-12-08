import os
import transformers
import adatest
from dotenv import load_dotenv

load_dotenv()

# create a HuggingFace sentiment analysis model
classifier = transformers.pipeline("sentiment-analysis", top_k=None)

# specify the backend generator used to help you write tests
generator = adatest.generators.OpenAI('curie', api_key=os.environ.get('OPENAPI_API_KEY'))

# ...or you can use an open source generator
#neo = transformers.pipeline('text-generation', model="EleutherAI/gpt-neo-125M")
#generator = adatest.generators.Transformers(neo.model, neo.tokenizer)

# create a new test tree
tests = adatest.TestTree("./output/Nimble_Chat_Bot.csv")

# adapt the tests to our model to launch a notebook-based testing interface
# (wrap with adatest.serve to launch a standalone server)
adatest.serve(tests.adapt(classifier, generator, auto_save=True))