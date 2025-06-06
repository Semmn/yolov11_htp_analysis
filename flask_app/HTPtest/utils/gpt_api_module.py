from openai import OpenAI

def generate_htp_recommendations(htp_analysis):
    client = OpenAI(
        api_key="Enter Your API Key Here!" 
    )

    messages = [
        {
            'role': 'system',
            'content': '당신은 HTP 검사 결과를 기반으로 환자에게 심리적으로 도움이 되는 실생활 중심의 조언을 제공하는 심리상담 전문가입니다. 사용자는 은둔형 외톨이로 병원이나 외부 활동이 어렵고, 치료에 대한 부담감도 큽니다.'
        },
        {
            'role': 'user',
            'content': f"""
아래는 집, 나무, 사람1, 사람2에 대한 심리 분석 결과입니다. 이 분석 내용을 기반으로, **사용자가 일상생활에서 실천할 수 있는 간단하고 쉬운 조언만을 종합적으로 정리해 주세요**.

분석:
1. 집: {htp_analysis[0]}
2. 나무: {htp_analysis[1]}
3. 사람1: {htp_analysis[2]}
4. 사람2: {htp_analysis[3]}

요구사항:
- '분석 결과'와 '추천 조언'은 각각 5~8문장으로 하나의 문단으로 요약해주세요.
- 병원이나 전문 치료, 상담, 정신과 등의 단어는 사용하지 마세요.
- 사용자는 집에만 있는 사람입니다. 실내에서 조용히 해볼 수 있는 활동, 감정 안정, 자기이해에 도움이 되는 습관, 간단한 루틴 등을 제안해주세요.
- 부담 없이 실천할 수 있도록 말투는 부드럽고 편안하게 해주세요.
- 조언은 문장 위주로 5~8문장 정도로 간결하게 써주세요.
"""
        }
    ]

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        store=True,
        messages=messages
    )

    return completion.choices[0].message.content
