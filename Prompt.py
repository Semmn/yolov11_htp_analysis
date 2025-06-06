# 첫번째 이미지는 원래 입력 이미지, 두번째 이미지는 detector로 추출한 이미지
# OpenAI API에서 이미지 입력이 여러개 일때 서로 구분하는 방법은, 먼저 입력한 이미지부터 서수를 붙이는 것이다.

SYSTEM_PROMPT = """You are an analyzer who accurately and professionally analyzes the images given for the HTP test. You answer user"s questions accurately and confidently, and do not give unnecessary answers other than the user"s questions."""
USER_PROMPT= {
    "집": {
        '지붕': """The given first image is the image of the whole house and the given second image is the image of only the roof extracted from the whole house image. For each of the four questions, answer only 'yes' or 'no' if applicable.
                            
                                The coordinates of the roof extracted from the whole image are: [{}, {}, {}, {}] in (x, y, w, h) format.
                                
                                User Questions:
                                1. Is the roof in the given second image in a state of collapse and cracks?
                                2. Are the roof and tiles depicted in detail in the given second image?
                                3. In the given second image, is the roof drawn with blurry and weak lines?
                                4. In the given second image, is the roof reinforced with strong pen pressure or has a repetitive outline?

                                Answers:
                                1: (answer here)
                                2: (answer here)
                                3: (answer here)
                                4: (answer here)""",
        
        '집벽': """The given first image is the image of the whole house and the given second image is the image of only the house wall extracted from the whole house image. For each of the five questions, answer only 'yes' or 'no' if applicable.
                            
                                The coordinates of the house wall extracted from the whole image are: [{}, {}, {}, {}] in (x, y, w, h) format.
                                
                                User Questions:
                                1. Is the given second image shows a solidly drawn wall?
                                2. Is the given second image shows a wall drawn with weak and powerless lines?
                                3. Does the given first image have a wall that shows the interior of the house through the wall?
                                4. In the given second image, does it have both the ground line of the wall and the door?
                                5. Does the given seonc image emphasize the ground line of the wall?

                                Answers:
                                1: (answer here)
                                2: (answer here)
                                3: (answer here)
                                4: (answer here)
                                5: (answer here)""",
        
        '문': """The given first image is the image of the whole house and the given second image is the image of only the door extracted from the whole house image. For each of the four questions, answer only 'yes' or 'no' if applicable.
                            
                                The coordinates of the door extracted from the whole image are: [{}, {}, {}, {}] in (x, y, w, h) format.
                                
                                User Questions:
                                1. Is the door open in the given second image?
                                2. Is the door locked in the given second image? (If it is locked, the door is completely closed and there is a lock, etc.)
                                3. In the second image, does the door have a lock or hinges?
                                4. Does the given first image contains a double door?

                                Answers:
                                1: (answer here)
                                2: (answer here)
                                3: (answer here)
                                4: (answer here)""",
        '창문': """The given first image is the image of the whole house and the given second image is the image of only the window extracted from the whole house image. For each of the six questions, answer only 'yes' or 'no' if applicable.
                            
                                The coordinates of the window extracted from the whole image are: [{}, {}, {}, {}] in (x, y, w, h) format.
                                
                                User Questions:
                                1. Are the windows locked in the given first image?
                                2. Are there curtains or shutters on the windows in the given first image?
                                3. Does the given first image have many windows with grids?
                                4. Does the given image emphasize the window frame?
                                5. In the given first image, are the windows divided into two vertical lines or are they triangular?
                                6. Are there any semicircular or circular windows in the given first image?

                                Answers:
                                1: (answer here)
                                2: (answer here)
                                3: (answer here)
                                4: (answer here)
                                5: (answer here)
                                6: (answer here)""",
        '연기': """The given first image is the image of the whole house and the given second image is the image of only the smoke extracted from the whole house image. For each of the three questions, answer only 'yes' or 'no' if applicable.
                            
                                The coordinates of the smoke extracted from the whole image are: [{}, {}, {}, {}] in (x, y, w, h) format.
                                
                                User Questions:
                                1. Is there a chimney in the picture with thick smoke coming out of it in given first picture?.
                                2. Is there a sinel thin line of light smoke in the given second image?
                                3. Is there smoke flowing from right to left in the given first image?

                                Answers:
                                1: (answer here)
                                2: (answer here)
                                3: (answer here)""",
        '기타': """The given image is the image of the whole house. For each of the five questions, answer only 'yes' or 'no' if applicable.
                                
                                User Questions:
                                1. Does the given first image show a sidewalk or a walking path?
                                2. Are there any shrubs and flowers in the house drawing in the given first picture?
                                3. Is there a tree pointing to a house in the given first picture?
                                4. Are there any strong shadows on the eaves in the given first image?
                                5. Are the mountains depicted in detail in the given first image?

                                Answers:
                                1: (answer here)
                                2: (answer here)
                                3: (answer here)
                                4: (answer here)
                                5: (answer here)"""
        },
    "나무": {
        "줄기": """The given first image is the image of the whole tree. For each of the twelve questions, answer only 'yes' or 'no' if applicable.
                                
                                User Questions:
                                1. In the given first image, does the tree trunk taper more abruptly toward the leaves?
                                2. In the given first image, do the ends of the tree trunks appear thicker?
                                3. Are the tree trunks noticeably thick in the given first image?
                                4. Is the tree trunk curved in the given first image?
                                5. In the given first image, are the tree trunks of uniform thickness like utility poles?
                                6. Is the root aprt of the stem overemphasized in the given first image?
                                7. Are the tree trunks swaying in the wind in the given first image?
                                8. Are there any wounds on the tree trunk in the given first image?
                                9. Does the tree in the given first image have a wide trunk?
                                10. Is the tree trunk narrow in the given first image?
                                11. In the given first image, does the tree trunk extend from the top?
                                12. In the given first image, does the tree have no trunk?

                                Answers:
                                1: (answer here)
                                2: (answer here)
                                3: (answer here)
                                4: (answer here)
                                5: (answer here)
                                6: (answer here)
                                7: (answer here)
                                8: (answer here)
                                9: (answer here)
                                10: (answer here)
                                11: (answer here)
                                12: (answer here)""",
        
        "가지": """The given first image is the image of the whole tree. For each of the twelve questions, answer only 'yes' or 'no' if applicable.
               
                                The coordinates of the branches extracted from the whole image are: [{}, {}, {}, {}] in (x, y, w, h) format.
                                
                                User Questions:
                                1. In the given first image, are the branches branching off from the trunk drawn in increasingly detailed ways?
                                2. Does the tree in the given first image have branches that get thicker towards the end?
                                3. Does the tree in the given image have single branches extending straight from the trunk?
                                4. In the given first image, the tree has branches that are forked and the tips of the branches are not closed at all?
                                5. In the given first image, are the tips of the tree branches depicted as sharp like spear points, or like thorns attached to the trunk?
                                6. In the given first image, does the tree have branches that extend upward rather than sideways?
                                7. In the given first image, does the tree have a configuration where its branches are not spread out very much?
                                8. In the given first image, are the branches of the tree noticeably larger than the trunk?
                                9. In the given first image, does the tree have small branches compared to the trunk?
                                10. In the given first image, are the branches of the tree facing the sun??
                                11. In the given first image, does the tree appear severly assymmetrical toward the house?
                                12. In the given first image, is the tree far from the house?

                                Answers:
                                1: (answer here)
                                2: (answer here)
                                3: (answer here)
                                4: (answer here)
                                5: (answer here)
                                6: (answer here)
                                7: (answer here)
                                8: (answer here)
                                9: (answer here)
                                10: (answer here)
                                11: (answer here)
                                12: (answer here)""",
        
        "뿌리": """The given first image is the image of the whole tree and the given second image is the image of only the roots extracted from the whole tree image. For each of the four questions, answer only 'yes' or 'no' if applicable.
                            
                                The coordinates of the root extracted from the whole image are: [{}, {}, {}, {}] in (x, y, w, h) format.
                                
                                User Questions:
                                1. In the given second image, are the roots clearly dried and dead?
                                2. Is the picture drawn so that the roots are visible through the ground?
                                3. Are the roots drawn at the edge of the drawing paper? (root must be located in the corner of the first image clearly)
                                4. Are the tree roots emphasized in the images?

                                Answers:
                                1: (answer here)
                                2: (answer here)
                                3: (answer here)
                                4: (answer here)""",
        
        '잎': """The given first image is the image of the whole tree and the given second image is the image of only the leaf extracted from the whole tree image. For each of the four questions, answer only 'yes' or 'no' if applicable.
                            
                                The coordinates of the leaf extracted from the whole image are: [{}, {}, {}, {}] in (x, y, w, h) format.
                                
                                User Questions:
                                1. Are there any leaves drawn on the crown in the given second image?
                                2. Are there any leaves falling from the tree in the given first image?
                                3. In the given first image, does the tree have leaves that are disproportionately large compared to its branches?
                                4. In the given image, is the shape of leaves similar to the shape of hands?

                                Answers:
                                1: (answer here)
                                2: (answer here)
                                3: (answer here)
                                4: (answer here)""",
        
        "기타": """The given first image is the image of the whole tree. For each of the six questions, answer only 'yes' or 'no' if applicable.
                                
                                User Questions:
                                1. Does the given first image depict a tree with flowers on it?
                                2. In the given first image, are fruits falling from the tree?
                                3. Does the sun exist in the given first image?
                                4. Are there any clouds between the trees and the sum in the given first image?
                                5. In the given first image, does the tree look like a keyhole?
                                6. Is there a squirrel on the tree in the given first image?

                                Answers:
                                1: (answer here)
                                2: (answer here)
                                3: (answer here)
                                4: (answer here)
                                5: (answer here)
                                6: (answer here)""",
        },
    
    
    "사람": {
        '얼굴': """The given first image is the image of the whole person and the given second image is the image of only the face extracted from the whole person image. For each of the three questions, answer only 'yes' or 'no' if applicable.
                            
                                The coordinates of the face extracted from the whole image are: [{}, {}, {}, {}] in (x, y, w, h) format.
                                
                                User Questions:
                                1. In the given image, is the person's face depicted as the back of the head?
                                2. In the given image, is the person's face drawn in profile?
                                3. Does the person in the given image have a beard on his face?

                                Answers:
                                1: (answer here)
                                2: (answer here)
                                3: (answer here)""",
        
        '눈': """The given first image is the image of the whole person and the given second image is the image of only the eye extracted from the whole person image. For each of the nine questions, answer only 'yes' or 'no' if applicable.
                            
                                The coordinates of the eye extracted from the whole image are: [{}, {}, {}, {}] in (x, y, w, h) format.
                                
                                User Questions:
                                1. In the given first image, does the person have a single eye?
                                2. In the given first image, is the person's eyes covered by hair or a hat?
                                3. In the given first image, is the person's face depicted with the eyes emphasized?
                                4. Does the given first image have an outline of the eyes without pupils on the human face?
                                5. In the given first image, are the human eyes represented by thin lines or dots?
                                6. Does the given first image depict any human eyes with eyelids or eyelashes?
                                7. In the given first image, are the eyebrows of the human eyes in the given image well-aligned?
                                8. Does the person in the given first image have upward-turned eyebrows?
                                9. Are the person's eyebrows depicted as dark in the given first image?

                                Answers:
                                1: (answer here)
                                2: (answer here)
                                3: (answer here)
                                4: (answer here)
                                5: (answer here)
                                6: (answer here)
                                7: (answer here)
                                8: (answer here)
                                9: (answer here)""",
        
        '귀': """The given first image is the image of the whole person and the given second image is the image of only the ear extracted from the whole person image. For each of the two questions, answer only 'yes' or 'no' if applicable.
                            
                                The coordinates of the ear extracted from the whole image are: [{}, {}, {}, {}] in (x, y, w, h) format.
                                
                                User Questions:
                                1. Are the person's ears very emphasized in the given first image?
                                2. Does the person in the given first image have an earring on his ear?
                                
                                Answers:
                                1: (answer here)
                                2: (answer here)""",
        
        '코': """The given first image is the image of the whole person and the given second image is the image of only the mouth extracted from the whole person image. For each of the seven questions, answer only 'yes' or 'no' if applicable.
                            
                                The coordinates of the nose extracted from the whole image are: [{}, {}, {}, {}] in (x, y, w, h) format.
                                
                                User Questions:
                                1. Does the person in the given second image have a triangule shaped nose?
                                2. Does the person in the given second image have a long nose?
                                3. Does the person in the given second image have a sharp nose?
                                4. Does the person in the given second image have a Roman nose?
                                5. Are the person's nostrils highlighted in the given first image?
                                6. In the given first image, is the person's nose drawn as being large?
                                7. Does the person in the given second image have a shaded nose?
                                
                                Answers:
                                1: (answer here)
                                2: (answer here)
                                3: (answer here)
                                4: (answer here)
                                5: (answer here)
                                6: (answer here)
                                7: (answer here)""",
        
        '입': """The given first image is the image of the whole person and the given second image is the image of only the mouth extracted from the whole person image. For each of the four questions, answer only 'yes' or 'no' if applicable.
                            
                                The coordinates of the mouth extracted from the whole image are: [{}, {}, {}, {}] in (x, y, w, h) format.
                                
                                User Questions:
                                1. Is the person in the given first image having a cynical sneer?
                                2. Does the person in the given first image have a smiling mouth?
                                3. Does the person in the given image have an open mouth?
                                4. Any there any teeth drawn along with a human mouth in the given second image?
                                
                                Answers:
                                1: (answer here)
                                2: (answer here)
                                3: (answer here)
                                4: (answer here)""",
        
        '턱': """The given first image is the image of the whole person. For each of the four questions, answer only 'yes' or 'no' if applicable.
                            
                                
                                User Questions:
                                1. Does the person in the given first image have a strong, prominent jaw?
                                2. Does the person in the given first image have a weak chin?
                                3. Does the person in the given first image have a split line on their chin?
                                4. Does the person in the given first image have a beard?
                                
                                Answers:
                                1: (answer here)
                                2: (answer here)
                                3: (answer here)
                                4: (answer here)""",
        
        '머리카락': """The given first image is the image of the whole person and the given second image is the image of only the hair extracted from the whole person image. For each of the six questions, answer only 'yes' or 'no' if applicable.
                            
                                The coordinates of the hair extracted from the whole image are: [{}, {}, {}, {}] in (x, y, w, h) format.
                                
                                User Questions:
                                1. Does the person in the given second image have untidy hair?
                                2. Does the given second image emphasize exaggerated detail in the human hair?
                                3. Does the person in the given second image have very thick and dark hair?
                                4. Does the person in the given second image have too little hair?
                                5. Does the person in the given second image have messy or untidy hair?
                                6. Does the given first image depict the human hair with much attention to detail?
                                
                                Answers:
                                1: (answer here)
                                2: (answer here)
                                3: (answer here)
                                4: (answer here)
                                5: (answer here)
                                6: (answer here)""",
    }
}