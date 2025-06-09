# detect_analysis_gpt_module.py
from HTPtest.utils.HTPTest_module import HTPTest
from HTPtest.utils.gpt_api_module import generate_htp_recommendations
import os

def run_detect_analysis_gpt_advice(session_id):
    session_dir = os.path.join('static', 'uploads', session_id)
    

    htp_test = HTPTest(
        person_name=session_id,
        person_age='25',
        drawing_time=[5, 8, 10],
        cls_config_file='./HTPtest/configs/child_painting.yaml',
        api_model='gpt-4.1-mini',
        paper_type='A4'
    )
    
    htp_test.create_client(api_key='Enter Your API!')

    htp_test.load_img([
        os.path.join(session_dir, 'house.jpg'),
        os.path.join(session_dir, 'tree.jpg'),
        os.path.join(session_dir, 'person1.jpg'),
        os.path.join(session_dir, 'person2.jpg')
    ])

    htp_test.run_detector(weight_path='./HTPtest/weights/best.pt')
    htp_analysis = htp_test.HTP_analysis()
    recommendations = generate_htp_recommendations(htp_analysis)

    return htp_analysis, recommendations
