import os
# [중요] Windows 환경에서 torch DLL 로드 오류(WinError 1114) 해결을 위한 설정
# 이 설정은 Windows에서 torch 라이브러리 로딩 충돌을 방지합니다.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, jsonify
from flask_cors import CORS 

# run_model.py에서 분석 함수와 첨삭 함수를 모두 가져옵니다.
# 모듈화를 통해 비즈니스 로직(run_model)과 서버 로직(app)을 분리했습니다.
from run_model import get_ai_feedback, call_external_llm_for_rewrite

app = Flask(__name__)

# [CORS 설정]
# 보안을 위해 프론트엔드 개발 서버(localhost:3000)에서의 API 요청만 허용합니다.
CORS(app, resources={r'/api/*': {'origins': 'http://localhost:3000'}})

@app.route('/api/revise', methods=['POST'])
def revise_essay():
    """
    [메인 API] 자소서 첨삭 요청 처리
    Input: JSON { "essay_text": "...", "job_description": "..." }
    Output: JSON { "feedback_result": "...", "feedback_rewrite": "...", ... }
    """
    try:
        data = request.get_json() 
        user_essay = data.get('essay_text')
        job_desc = data.get('job_description')

        # 유효성 검사
        if not user_essay or not job_desc:
            return jsonify({'error': '자소서(essay_text)와 채용공고(job_description)가 모두 필요합니다.'}), 400

        print(f"📥 받은 자소서 길이: {len(user_essay)}자")
        print(f"📥 받은 공고 길이: {len(job_desc)}자")

        try:
            # ---------------------------------------------------------
            # [Process 1] 로컬 AI 모델 분석
            # 역할: Ko-Qwen2 모델이 공고와 자소서를 분석해 점수와 피드백(JSON) 생성
            # ---------------------------------------------------------
            print("🤖 1단계: 로컬 AI 분석 시작...")
            ai_feedback_json_string = get_ai_feedback(job_desc, user_essay)
            print("✅ 1단계 분석 완료")
            
            # ---------------------------------------------------------
            # [Process 2] 외부 AI(Gemini) 자소서 첨삭
            # 역할: 1단계 분석 결과를 바탕으로 Gemini가 자소서를 전문적으로 다시 작성
            # ---------------------------------------------------------
            print("✨ 2단계: Gemini 자소서 첨삭 시작...")
            ai_rewrite_string = call_external_llm_for_rewrite(ai_feedback_json_string, user_essay)
            print("✅ 2단계 첨삭 완료")
            
            # ---------------------------------------------------------
            # [Final] 결과 반환
            # ---------------------------------------------------------
            return jsonify({
                'feedback_result': ai_feedback_json_string, # 구조적 분석 (JSON 문자열)
                'feedback_rewrite': ai_rewrite_string,      # 첨삭/조언 (일반 텍스트)
                'cheer_message': "AI 분석과 첨삭이 완료되었습니다! 합격을 기원합니다."
            })
            
        except Exception as ai_e:
            print(f"🔥 모델 처리 중 오류 발생: {ai_e}")
            return jsonify({'error': f'AI 모델 처리 중 오류: {ai_e}'}), 500

    except Exception as e:
        print(f"🔥 서버 내부 오류 발생: {e}")
        return jsonify({'error': '서버 처리 중 오류가 발생했습니다.'}), 500

if __name__ == '__main__':
    # [서버 실행]
    # AI 모델 로딩은 메모리를 많이 사용하므로 debug=False로 실행 (중복 로드 방지)
    # host='0.0.0.0' 설정으로 외부 접속 허용 (필요시)
    app.run(host='0.0.0.0', debug=False, port=5000)

