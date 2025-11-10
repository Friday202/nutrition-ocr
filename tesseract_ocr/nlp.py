from transformers import pipeline


def extract_nutrition_info(question, context):
    qa_pipeline = pipeline("question-answering")
    result = qa_pipeline(question=question, context=context)
    return result['answer']


def extract_nutrition_info_generative(question, context):
    qa_pipeline = pipeline("text2text-generation")
    input_text = f"question: {question} context: {context}"
    result = qa_pipeline(input_text)
    return result[0]['generated_text']