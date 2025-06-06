# https://blog.naver.com/wink4783/220022515082 -> HTP 투사검사 질문지 및 해석
# https://m.blog.naver.com/sswan77/222254918861 -> HTP 투사검사 해석

import yaml
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

'''
yaml_file = './configs/child_painting.yaml'
with open(yaml_file, 'r') as yaml_read:
    configs = yaml.load(yaml_read, Loader=yaml.FullLoader)
idx2cls = configs['names']
cls2idx = {v: k for k, v in idx2cls.items()}
'''
# HTPTest class
class HTPTest:
    def __init__(self, person_name, person_age, drawing_time, cls_config_file, paper_type='A4'):
        self.person_name = person_name # 사람 이름
        self.person_age = person_age # 사람 나이
        self.drawing_time = drawing_time # 각 HTP (House, Tree, Person1, Person2(반대 성으로))에 소요된 시간
        self.total_drawing_t = sum(drawing_time) # 총 그리는데 걸리는 시간
        self.img_loaded = False # load_img 호출시 True
        
        self.paper_type = paper_type # 사용할 종이 종류
        self._paper_size_cm = {
            'A4': (21, 29.7), # width, height
        }
        self.detector_ran = False # run_detector 호출시 True
        with open(cls_config_file, 'r', encoding='utf-8') as fr:
            self.cls_config = yaml.load(fr, Loader=yaml.FullLoader) # config settings for yolo classification
        self.idx2cls = self.cls_config['names'] # 자세한 클래스 이름 및 인덱스 배정은 child_painting.yaml 파일을 참조
        self.cls2idx = {v: k for k, v in self.idx2cls.items()}
        self.htp_imgs = []
        
    def load_img(self, imgs, direct_input=False):
        # direct_input: True인 경우 이미지 객체를 (PIL형태를 따름) 직접 입력, False인 경우 이미지 경로를 기반으로 로드
        # direct_input=True인 경우 각 이미지들은 python 배열로 둘러싸여 있어야 함
        if direct_input:
            self.htp_imgs = imgs # 이미지 배열 (HTP 검사 순서에 따라, House -> Tree -> Person1 -> Person2 순서를 가짐)
        else:
            for img in imgs:
                self.htp_imgs.append(np.asarray(Image.open(img))) # numpy 객체 형태로 전환
        
        self.img_loaded=True
    
    def clear(self):
        self.img_loaded=False
        self.htp_imgs = []
        self.drawing_time = None
        self.total_drawing_t = None
        self.person_name = ""
        
    def update_img(self, htp_type, img, direct_input=False):
        # htp_type: 'house': 0, 'tree': 1, 'person1': 2, 'person2': 3
        if not self.img_loaded:
            raise Exception("Image Must be Loaded First!")
        if not direct_input:
            img = np.asarray(Image.open(img))
        self.htp_imgs[htp_type] = img
        
    def run_detector(self, weight_path):
        trained_yolo = YOLO(weight_path)
        self.detected_result = []
        for img in self.htp_imgs:
            self.detected_result.append(trained_yolo(img))
        
        self.detector_ran = True
        
    # A4 사이즈 용지에서 수행한다고 가정하면 -> 21 x 29.7 cm
    def HTP_analysis(self):
        
        if not self.img_loaded:
            raise Exception("Image Must be Loaded First!")
        if not self.detector_ran:
            raise Exception("Inference Detector First!")
        
        htp_analysis = [] # list of strings that contains analysis result
        htp_type = ['house', 'tree', 'person1', 'person2']
        
        for htp, results in zip(htp_type, self.detected_result):
            analysis = []
            for i in range(len(results)):
                result = results[i]
                img_shape = result.orig_shape # (height, width)
                boxes = result.boxes
                box_cls = boxes.cls.int().detach().cpu().numpy()
                box_loc = boxes.xywhn.detach().cpu().numpy()
                box_loc_p = boxes.xywh.detach().cpu().numpy()
                box_area = box_loc[:, 2] * box_loc[:, 3]
                num_boxes = len(boxes) # number of boxes
                
                if htp == 'person1' or htp == 'person2':
                    person_xywhn = box_loc[box_cls==self.cls2idx['사람전체']]
                    if len(person_xywhn) >= 1:
                        person_xywhn = person_xywhn[np.argmax(person_xywhn[:, 2] * person_xywhn[:, 3])] # 가장 큰 집 bbox를 선택
                    else:
                        print("Person BBoxes are not found on the image!")
                        raise Exception("There is no any Person detected. Please resubmit the image with Person drawn on it.")
                    person_area = person_xywhn[2] * person_xywhn[3] # area of the person bbox
                    
                    # 아주 크게 그리거나 작게 그린 경우 (용지에 80% 이상을 사용하였거나, 10%미만 사용했을 경우) 또한 전체 object 갯수가 3개 이하의 공허하게 그려진 경우
                    if (person_area >= 0.8 or person_area < 0.1) and num_boxes <= 3:
                        analysis.append(['정서장애'])
                    # 보통 사이즈보다 조금 더 작은 경우
                    elif person_area >= 0.15 and person_area <= 0.3:
                        # 도화지의 상부에 그려진 경우
                        if person_xywhn[1] <= 0.3:
                            analysis.append(['열등감, 무능력함, 억제, 소심, 낮은 에너지 수준, 이치에 맞지 않는 낙천주의'])
                        else:
                            analysis.append(['열등감, 무능력함, 억제, 소심'])
                    # 보통 사이즈보다 아주 더 작은 경우
                    elif person_area < 0.15:
                        analysis.append(['수축된 자아, 심한 우울증'])
                    # 보통 사이즈보다 아주 큰 경우
                    elif person_area >= 0.8:
                        analysis.append(['조증환자, 아동정신적 장애, 기절적 장애'])
                    
                    # 인물화에 집이 포함되어 있는경우
                    if (box_cls==self.cls2idx['집전체']).any():
                        mask = box_cls==self.cls2idx['집전체']
                        box_house = box_loc[mask]
                        max_house_size = np.max(box_house[:, 2] * box_house[:, 3]) # 최대 집 크기
                        if person_area - max_house_size >= 0.25:
                            analysis.append(['가족보다는 자신을 더 중요시한다는 경향'])
                    
                    # 머리 크기에 따른 분석
                    head_xywhn = box_loc[box_cls==self.cls2idx['머리']]
                    head_xywhn = head_xywhn[np.argmax(head_xywhn[:, 2] * head_xywhn[:, 3])] if len(head_xywhn) > 0 else None
                    if head_xywhn is not None:
                        head_area = head_xywhn[2] * head_xywhn[3]
                        if head_area / person_area < 0.12:
                            analysis.append(['강박증 환자들, 지적 부족감'])
                        elif head_area / person_area >= 0.33:
                            analysis.append(['아동, 강한 지적 노력, 지적 성취에 대한 압박, 공격성, 자기중심적인 태도, 편집증'])
                    else:
                        analysis.append(['사고 장애, 신경학적 장애, 물건이나 모자 등에 머리가 다 가려지게 그리는 경우 자신의 지적 능력에 자신감이 없고 불안감을 느낌'])
                    
                    # 눈의 크기에 따른 분석    
                    eye_xywhn = box_loc[box_cls==self.cls2idx['눈']]
                    eye_xywhn = eye_xywhn[np.argmax(eye_xywhn[:, 2] * eye_xywhn[:, 3])] if len(eye_xywhn) > 0 else None
                    if eye_xywhn is not None:
                        eye_area = eye_xywhn[2] * eye_xywhn[3]
                        if eye_area / person_area < 0.005:
                            analysis.append(['사회적 상호작용에서 위축되고 회피하고자 함. 자아도취'])
                        elif eye_area / person_area > 0.02:
                            analysis.append(['타인과 정서적 교류에 있어서 지나치게 예민함'])
                        elif eye_area / person_area >= 0.01 and eye_area / person_area <= 0.02:
                            analysis.append(['감정적 교류에 있어서 불안감과 긴장감 타인과의 상호작용에서 의심이나 방어적 태도, 편집증적인 경향성'])
                    else:
                        analysis.append(['타인과 감정을 교류하는데 극심한 불안감을 느낌 사고장애의 가능성'])
                        
                    # 귀의 크기에 따른 분석
                    ear_xywhn = box_loc[box_cls==self.cls2idx['귀']]
                    ear_xywhn = ear_xywhn[np.argmax(ear_xywhn[:, 2] * ear_xywhn[:, 3])] if len(ear_xywhn) > 0 else None
                    if ear_xywhn is not None:
                        ear_area = ear_xywhn[2] * ear_xywhn[3]
                        if ear_area / person_area < 0.0020:
                            analysis.append(['정서적 자극을 피하고 싶고 위축되어 있음'])
                        elif ear_area / person_area >= 0.01:
                            analysis.append(['대인관계 상황에서 너무 예민함'])
                    else:
                        analysis.append(['아동, 정서적 교류나 감정표현에 대해 불안하고 자신이 없어함'])
                        
                    # 코의 크기에 따른 분석
                    nose_xywhn = box_loc[box_cls==self.cls2idx['코']]
                    nose_xywhn = nose_xywhn[np.argmax(nose_xywhn[:, 2] * nose_xywhn[:, 3])] if len(nose_xywhn) > 0 else None
                    if nose_xywhn is not None:
                        nose_area = nose_xywhn[2] * nose_xywhn[3]
                        if nose_xywhn[2] / nose_xywhn[3] >= 3:
                            analysis.append(['공격성, 우월을 탐한다. 외형적이고 활동적이다.'])
                        if nose_area / person_area >= 0.01:
                            analysis.append(['초기의 우울증 환자가 그리는 경향 자신의 남성역활을 하는 것을 부정하는 사람도 큰 코를 그리는 경향이 있음'])
                    else:
                        analysis.append(['성에 대해 무엇인가 갈등이 있으며 남성적인 것을 거부하며 거세불안이 있고 동성애 경향이 있을 가능성이 있음 타인에게 어떻게 보일지에 매우 예민하고 두려워 함'])
                        
                    # 입의 크기에 따른 분석
                    mouth_xywhn = box_loc[box_cls==self.cls2idx['입']]
                    mouth_xywhn = mouth_xywhn[np.argmax(mouth_xywhn[:, 2] * mouth_xywhn[:, 3])] if len(mouth_xywhn) > 0 else None    
                    if mouth_xywhn is not None:
                        mouth_area = mouth_xywhn[2] * mouth_xywhn[3]
                        if mouth_area / person_area > 0.05: # 입이 너무 큼
                            analysis.append(['타인과의 정서적 교류, 애정의 교류에 있어서 불안감을 느끼지만 과도하게 적극적이고 주장적이고 심지어 공격적인 태도를 취함으로써 역공포적으로 이러한 불안감을 보상받으려 함'])
                        elif mouth_area / person_area < 0.0020: # 입의 너무 작은
                            analysis.append(['내적인 상처를 받지 않으려고 정서적 상호작용을 회피하거나 타인의 애정어린 태도를 거절하고자 함, 이와 관련하여 절망감이나 우울감을 느낌'])
                        elif mouth_xywhn[3] * img_shape[1] < 5 and mouth_xywhn[2] * img_shape[0] > 10: # 가로선 하나만 존재하는 경우
                            analysis.append(['타인과의 정서적 교류에서 무감각, 냉정한 태도'])                    
                    else:
                        analysis.append(['애정 욕구의 강한 거부, 심한 죄의식, 천식환자, 우울', '부모와 같은 대상과의 관계에 상당한 갈등이나 결핍이 있음'])
                        
                if htp == 'house':
                    house_xywhn = box_loc[box_cls==self.cls2idx['집전체']]
                    if len(house_xywhn) >= 1:
                        house_xywhn = house_xywhn[np.argmax(house_xywhn[:, 2] * house_xywhn[:, 3])] # 가장 큰 집 bbox를 선택
                    else:
                        print("House BBoxes are not found on the image!")
                        raise Exception("There is no any house detected. Please resubmit the image with house drawn on it.")
                    house_area = house_xywhn[2] * house_xywhn[3] # area of the house bbox
                    
                    if not (box_cls==self.cls2idx['문']).any():
                        analysis.append(['가정환경에서 타인과 접촉하지 않으려는 감정, 외부세계와의 교류를 원치 않는 냉정한 사람'])
                    
                    num_of_windows = (box_cls==self.cls2idx['창문']).sum()
                    if num_of_windows == 0: # 창문이 없는 경우 (생략된 경우)
                        analysis.append(['철회와 상당한 편집증적 경향성'])
                    elif num_of_windows >= 5: # 창문이 많은 경우 (5개 이상 창문을 그린 경우)
                        analysis.append(['개방과 환경적 접촉에 대한 갈망'])
                        
                    if (box_cls==self.cls2idx['울타리']).any():
                        analysis.append(['방어의 수단, 안전을 방해받고 싶지 않다는 것'])
                
                    if (box_cls==self.cls2idx['산']).any():
                        analysis.append(['어머니의 보호를 구하며 안전의 욕구를 지님'])
                        
                    if (box_cls==self.cls2idx['나무']).any():
                        num_of_trees = (box_cls==self.cls2idx['나무']).sum()
                        if num_of_trees >= 4: # 나무가 4그루 이상 심어진 경우
                            analysis.append(['방어벽을 만들려는 시도 만약 산책길과 연결된 경우, 어느정도 불안이 있으나 그것을 통제하고자 하는 의식적인 시도'])
                            
                    # 지붕의 선이나 디테일한 묘사에 대한 판단은 yolo로만 구현할 수 없음
                    roof_xywhn = box_loc[box_cls==self.cls2idx['지붕']]
                    roof_xywhn = roof_xywhn[np.argmax(roof_xywhn[:, 2] * roof_xywhn[:, 3])] if len(roof_xywhn) > 0 else None
                    if roof_xywhn is not None:
                        roof_area = roof_xywhn[2] * roof_xywhn[3] # area of the roof bbox
                        # 지붕이 너무 크게 그려진 경우 -> 전체 집 크기에 75% 이상을 차지하는 경우
                        if roof_area / house_area >= 0.75:
                            analysis.append(['공상에 열중하며 외면적인 대인관계로부터 도피하려는 경향'])
                        elif roof_area / house_area < 0.75 and roof_area / house_area >= 0.4:
                            analysis.append(['적절한 사고 활동을 통해 현실을 균형 있게 살고 있음'])
                    # 벽또한 디테일한 묘사를 바탕으로 분석하는 것은 yolo로만 구현할 수 없음
                    # 문의 크기에 따른 또는 존재 여부에 따라 분석할 수 있다. 
                    # 문의 열리거나, 닫힘, 이중 문 등은 yolo로만 구현할 수 없음
                    
                    door_xywhn = box_loc[box_cls==self.cls2idx['문']]
                    door_xywhn = door_xywhn[np.argmax(door_xywhn[:, 2] * door_xywhn[:, 3])] if len(door_xywhn) > 0 else None
                    if door_xywhn is not None:
                        door_area = door_xywhn[2] * door_xywhn[3]
                        if  door_area / house_area < 0.1:
                            analysis.append(['환경과의 접촉을 꺼리고 우유부단함에 지배되고 있음'])
                        elif door_area / house_area >= 0.01 and door_area / house_area <= 0.1:
                            analysis.append(['타인에 대한 과도한 의존심'])
                            
                    # 다른 그림에 대한 분석은 multi-modal 모델로 시도해 볼 수 있음.
                    # 창문의 크기에 따른 분석은 가능하나, 디테일한 창문의 형태는 알 수 없음으로 분석이 어렵다.
                    # 연기또한 제대로 분석하기 어렵습니다.    
                
                # htp의 나무를 분석하는 경우 -> 나무는 세세한 디테일을 보고 판단하는 경우가 많아, yolo만을 가지고 분석을 하는 것이 어렵다.
                # 필요시 chat-gpt의 multi-modal 기능을 활용하여 image-to-text 기능을 이용하여 분석하는 것도 가능하다.
                if htp in ['tree']:
                    tree_xywhn = box_loc[box_cls==self.cls2idx['나무전체']]
                    if len(tree_xywhn) >= 1:
                        tree_xywhn = tree_xywhn[np.argmax(tree_xywhn[:, 2] * tree_xywhn[:, 3])]
                    else:
                        print("Tree BBoxes are not found on the image!")
                        raise Exception("There is no any tree detected. Please resubmit the image with tree drawn on it.")
                    tree_area = tree_xywhn[2] * tree_xywhn[3] # area of the tree bbox
                                
                ratio_count = list()
                area_per_ratio = list()
                mask1 = np.logical_and(np.logical_and(box_loc[:, 0] >= 0.45, box_loc[:, 0] <= 0.55), np.logical_and(box_loc[:, 1] >= 0.45, box_loc[:, 1] <= 0.55))
                ratio_count.append(mask1.sum())
                area_per_ratio.append(box_area[mask1].sum())
                
                mask2 = np.logical_and(np.logical_and(box_loc[:, 1] > 0.25, box_loc[:, 1] < 0.75), np.logical_and(box_loc[:, 0] > 0.15, box_loc[:, 0] <= 0.25))
                ratio_count.append(mask2.sum())
                area_per_ratio.append(box_area[mask2].sum())
                
                mask3 = np.logical_and(np.logical_and(box_loc[:, 1] > 0.25, box_loc[:, 1] < 0.75), np.logical_and(box_loc[:, 0] >= 0.75, box_loc[:, 0] < 0.85))
                ratio_count.append(mask3.sum())
                area_per_ratio.append(box_area[mask3].sum())
                
                mask4 = np.logical_and(np.logical_and(box_loc[:, 0] > 0.25, box_loc[:, 0] < 0.75), np.logical_and(box_loc[:, 1] > 0.15, box_loc[:, 1] <= 0.25))
                ratio_count.append(mask4.sum())
                area_per_ratio.append(box_area[mask4].sum())
                
                mask5 = np.logical_and(np.logical_and(box_loc[:, 0] > 0.25, box_loc[:, 0] < 0.75), np.logical_and(box_loc[:, 1] >= 0.75, box_loc[:, 1] < 0.85))
                ratio_count.append(mask5.sum())
                area_per_ratio.append(box_area[mask5].sum())
                
                mask6 = np.logical_and(np.logical_and(box_loc[:, 0] > 0.15, box_loc[:, 0] <= 0.25), np.logical_and(box_loc[:, 1] > 0.15, box_loc[:, 1] <= 0.25))
                ratio_count.append(mask6.sum())
                area_per_ratio.append(box_area[mask6].sum())
                
                mask7 = np.logical_and(np.logical_and(box_loc[:, 0] >= 0.75, box_loc[:, 0] < 0.85), np.logical_and(box_loc[:, 1] > 0.15, box_loc[:, 1] <= 0.25))
                ratio_count.append(mask7.sum())
                area_per_ratio.append(box_area[mask7].sum())
                
                mask8 = np.logical_and(np.logical_and(box_loc[:, 0] > 0.15, box_loc[:, 0] <= 0.25), np.logical_and(box_loc[:, 1] >= 0.75, box_loc[:, 1] < 0.85))
                ratio_count.append(mask8.sum())
                area_per_ratio.append(box_area[mask8].sum())
                
                mask9 = np.logical_and(np.logical_and(box_loc[:, 0] >= 0.75, box_loc[:, 0] < 0.85), np.logical_and(box_loc[:, 1] >= 0.75, box_loc[:, 1] < 0.85))
                ratio_count.append(mask9.sum())
                area_per_ratio.append(box_area[mask9].sum())
                
                mask10 = np.logical_or(np.logical_or(np.logical_or(box_loc[:, 0] >= 0.85, np.logical_and(box_loc[:, 0] <= 0.15, box_loc[:, 1] <= 0.15)),
                                       np.logical_and(box_loc[:, 0] <= 0.15, box_loc[:, 1] >= 0.85)), np.logical_or(np.logical_and(box_loc[:, 0] >= 0.85, box_loc[:, 1] <= 0.15), 
                                                                                                                    np.logical_and(box_loc[:, 0] >= 0.85, box_loc[:, 1] >= 0.85)))
                ratio_count.append(mask10.sum())
                area_per_ratio.append(box_area[mask10].sum())
                        
                # [중앙, 왼쪽, 오른쪽, 위쪽, 아래쪽, 왼쪽 위, 오른쪽 위, 왼쪽 아래, 오른쪽 아래, 극단적 위치 (너무 오른쪽, 아니면 코너)]
                max_ratio_index = np.argmax(area_per_ratio)
                total_numbers_of_bbox = np.sum(ratio_count)
                # 중앙에 위치한 bbox들 크기의 총합이 가장 크거나, 오른쪽, 왼쪽에 위치한 bbox 영역의 합이 서로 비슷한 경우
                if max_ratio_index == 0 or ((np.abs(area_per_ratio[2] - area_per_ratio[1]) < 0.15) and (np.abs(area_per_ratio[3] - area_per_ratio[4]) < 0.15)): 
                    analysis.append(['자기중심적', '불안정한 경우 스스로 통제하여 마음의 안정을 유지하려는 사람'])
                # 왼쪽에 위치한 bbox들의 크기가 가장 크면서, 오른쪽 보다 왼쪽에 bbox 영역이 치중된 경우 (15% 이상의 차이로)
                elif max_ratio_index == 1 and ((np.abs(area_per_ratio[2] - area_per_ratio[1])) > 0.15):
                    analysis.append(['자의식이 강하고 내향적 성향', '퇴행적상황, 공상적, 여성경향', '총동적으로 만족을 구하려는 경향성'])
                # 오른쪽에 위치한 bbox들의 크기가 가장 크면서, 왼쪽 보다 오른쪽으로 bbox 영역이 치중된 경우 (15% 이상의 차이로)
                elif max_ratio_index == 2 and ((np.abs(area_per_ratio[2] - area_per_ratio[1])) > 0.15):
                    analysis.append(['환경에 따라 방향을 결정하거나 미래강조, 남성적', '지적만족을 선호하는 사람', '인지적으로 감정을 지나치게 통제하거나 있거나 억제하는 경향'])
                # 위쪽인 경우도 마찬가지
                elif max_ratio_index == 3 and ((np.abs(area_per_ratio[3] - area_per_ratio[4]) > 0.15)):
                    analysis.append(['현실과 구체성, 확실성', '불안정으로 우울한 기분이거나 좌절을 갖고 있는 경우', '안정되고 침착한 경우'])
                # 아래쪽인 경우도 마찬가지
                elif max_ratio_index == 4 and ((np.abs(area_per_ratio[3] - area_per_ratio[4]) > 0.15)):
                    analysis.append(['높은 목표에 도달하려고 노력하며 때론 도달하기 어려운 공상에 만족하는 경향', '자기존재에 대한 불확실한 느낌', '낙천적이거나 자신이 타인으로부터 고립된 느낌'])
                elif max_ratio_index == 5:
                    analysis.append(['불안이 강하고 새로운 경험을 피하고 과거로의 퇴행'])
                elif max_ratio_index == 6:
                    analysis.append(['미래에 대한 과도적 낙관주의 성향이거나 미래지향적 환상'])
                elif max_ratio_index == 7:
                    analysis.append(['과거와 관련된 우울감'])
                elif max_ratio_index == 8:
                    analysis.append(['미래와 관련된 불안감'])
                elif max_ratio_index == 9:
                    analysis.append(['자신감이 없고 불안정감, 타인에게 지지받고자 하는 욕구', '의존경향, 독립에 대한 두려움을 반영할 수 있으며 새로운 경험에 대한 회피경향에 있거나 환상에 머물고 싶은 욕구'])
                # 한쪽으로 치중된 비율이 40% 미만이면서, 최소 5% 이상씩 모든 영역을 차지하는 경우 (용지 전체를 사용, 산만하게 그려진 경우)
                elif (np.array(ratio_count) / total_numbers_of_bbox).max() < 0.4 or (((np.array(ratio_count) / total_numbers_of_bbox)-0.05)>0).all():
                    analysis.append(['조증문제일 가능성'])
                
            htp_analysis.append(analysis)
        return htp_analysis