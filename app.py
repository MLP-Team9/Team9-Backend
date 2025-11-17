# app.py (수정본)

from flask import Flask, request, jsonify
from flask_cors import CORS 
from run_model import get_ai_feedback  # <--- [수정] 1. 'run_model.py'에서 AI 함수를 가져옵니다.

app = Flask(__name__)
CORS(app, resources={r'/api/*': {'origins': 'http://localhost:3000'}})

@app.route('/api/revise', methods=['POST'])
def revise_essay():
    try:
        data = request.get_json() 
        user_essay = data.get('essay_text')
        job_desc = data.get('job_description')

        if not user_essay or not job_desc:
            return jsonify({'error': '자소서(essay_text)와 채용공고(job_description)가 모두 필요합니다.'}), 400

        print(f"받은 자소서: {user_essay[:30]}...")
        print(f"받은 공고: {job_desc[:30]}...")

        # --- [수정] 2. 'mock_feedback' 대신 '진짜 AI' 호출 ---
        try:
            # 이 함수가 run_model.py의 AI를 실행하고 JSON 문자열을 반환합니다.
            ai_feedback_json_string = get_ai_feedback(job_desc, user_essay)
            
            print(f"AI 응답 (JSON 문자열): {ai_feedback_json_string}")
            
            # 프론트가 { "feedback_result": "..." } 형식을 기다리므로
            # AI가 생성한 JSON 문자열을 그대로 전달합니다.
            return jsonify({
                'feedback_result': ai_feedback_json_string 
            })
            
        except Exception as ai_e:
            print(f"AI 모델 추론 오류 발생: {ai_e}")
            return jsonify({'error': f'AI 모델 처리 중 오류: {ai_e}'}), 500
        # --- [수정 완료] ---

    except Exception as e:
        print(f"서버 오류 발생: {e}")
        return jsonify({'error': '서버 처리 중 오류가 발생했습니다.'}), 500

if __name__ == '__main__':
    # [수정] 3. AI 모델 로딩은 무거우므로, 'debug=True'를 끄고 실행하는 것을 권장합니다.
    # (debug=True는 파일이 수정될 때마다 서버를 재시작하는데, 
    # 그때마다 AI 모델(15GB)을 다시 로드해서 1~2분이 걸립니다)
    app.run(debug=False, port=5000)