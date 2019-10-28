#UI大改造v8
#-*- coding: utf8 -*-
import sys
from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
import urllib.request as req
import cv2
import numpy as np
import numpy
from imutils import face_utils
import dlib
import glob
import os
import matplotlib.pyplot as plt
PICTURE_HEIGHT=185
PICTURE_WIDTH=300
global rightface,leftface,frontface

# ボタンが押されたら呼び出される関数 Entryの中身をすべて削除します
def deleteEntry(self):
	root.destroy()
    # Entryの中身をすべて削除します
#ChangeYourFaceFunction()関係
detector = dlib.get_frontal_face_detector()
PREDICTOR_PATH = './shape_predictor_68_face_landmarks.dat'
PREDICTOR = dlib.shape_predictor(PREDICTOR_PATH)

#顔の向き初期設定↓！！！！！！！！！！
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])
reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])
line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]
#顔の向き初期設定終わり↑！！！！！！！！！！
def ChangeYourFaceFunction():
    """
    class NoFaces(Exception):
        pass
    """
    class Face:
        def __init__(self, image, rect):
            self.image = image
            self.landmarks = numpy.matrix(
            [[p.x, p.y] for p in PREDICTOR(image, rect).parts()]
            )
    class BeBean:
        SCALE_FACTOR = 1
        FEATHER_AMOUNT = 11

        # 特徴点のうちそれぞれの部位を表している配列のインデックス
        FACE_POINTS = list(range(17, 68))
        MOUTH_POINTS = list(range(48, 61))
        RIGHT_BROW_POINTS = list(range(17, 22))
        LEFT_BROW_POINTS = list(range(22, 27))
        RIGHT_EYE_POINTS = list(range(36, 42))
        LEFT_EYE_POINTS = list(range(42, 48))
        NOSE_POINTS = list(range(27, 35))
        JAW_POINTS = list(range(0, 17))

        ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS +
            NOSE_POINTS + MOUTH_POINTS)

        # オーバーレイする特徴点
        OVERLAY_POINTS = [LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
            NOSE_POINTS + MOUTH_POINTS]
        COLOR_CORRECT_BLUR_FRAC = 0.7
        def __init__(self, image_path, before_after = True):
            self.detector = dlib.get_frontal_face_detector()
            self._load_beans(image_path)
            self.before_after = before_after
        def load_faces_from_image(self, image_path):
            """
            画像パスから画像オブジェクトとその画像から抽出した特徴点を読み込む。
            ※ 画像内に顔が1つないし複数検出された場合も、返すので正確には「特徴点配列」の配列を返す
            """
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (image.shape[1] * self.SCALE_FACTOR,
                                    image.shape[0] * self.SCALE_FACTOR))
            rects = self.detector(image, 1)
            if len(rects) == 0:
                raise NoFaces
        #    else:出力がうるさいprint("Number of faces detected: {}".format(len(rects)))
            faces = [Face(image, rect) for rect in rects]
            return image, faces

        def transformation_from_points(self, t_points, o_points):
            """
            特徴点から回転やスケールを調整する。
            t_points: (target points) 対象の特徴点(入力画像)
            o_points: (origin points) 合成元の特徴点(つまりビーン)
            """
            t_points = t_points.astype(numpy.float64)
            o_points = o_points.astype(numpy.float64)
            t_mean = numpy.mean(t_points, axis = 0)
            o_mean = numpy.mean(o_points, axis = 0)
            t_points -= t_mean
            o_points -= o_mean
            t_std = numpy.std(t_points)
            o_std = numpy.std(o_points)
            t_points -= t_std
            o_points -= o_std
            # 行列を特異分解しているらしい
            # https://qiita.com/kyoro1/items/4df11e933e737703d549
            U, S, Vt = numpy.linalg.svd(t_points.T * o_points)
            R = (U * Vt).T

            return numpy.vstack(
            [numpy.hstack((( o_std / t_std ) * R, o_mean.T - ( o_std / t_std ) * R * t_mean.T )),
            numpy.matrix([ 0., 0., 1. ])]
            )

        def get_face_mask(self, face):
            image = numpy.zeros(face.image.shape[:2], dtype = numpy.float64)
            for group in self.OVERLAY_POINTS:
                self._draw_convex_hull(image, face.landmarks[group], color = 1)

            image = numpy.array([ image, image, image ]).transpose((1, 2, 0))
            image = (cv2.GaussianBlur(image, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0) > 0) * 1.0
            image = cv2.GaussianBlur(image, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0)

            return image

        def warp_image(self, image, M, dshape):
            output_image = numpy.zeros(dshape, dtype = image.dtype)
            cv2.warpAffine(
            image,
            M[:2],
            (dshape[1], dshape[0]),
            dst = output_image, borderMode = cv2.BORDER_TRANSPARENT, flags = cv2.WARP_INVERSE_MAP
            )
            return output_image

        def correct_colors(self, t_image, o_image, t_landmarks):
            """
            対象の画像に合わせて、色を補正する
            """
            blur_amount = self.COLOR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
            numpy.mean(t_landmarks[self.LEFT_EYE_POINTS], axis = 0) -
            numpy.mean(t_landmarks[self.RIGHT_EYE_POINTS], axis = 0)
            )
            blur_amount = int(blur_amount)

            if blur_amount % 2 == 0: blur_amount += 1

            t_blur = cv2.GaussianBlur(t_image, (blur_amount, blur_amount), 0)
            o_blur = cv2.GaussianBlur(o_image, (blur_amount, blur_amount), 0)

            # ゼロ除算を避ける　
            o_blur += (128 * (o_blur <= 1.0)).astype(o_blur.dtype)

            return (o_image.astype(numpy.float64) * t_blur.astype(numpy.float64) / o_blur.astype(numpy.float64))

        def to_bean(self, image_path):
            original, faces = self.load_faces_from_image(image_path)

            # base_imageに合成していく
            base_image = original.copy()

            for face in faces:
                bean = self._get_bean_similar_to(face)
                bean_mask = self.get_face_mask(bean)

                M = self.transformation_from_points(
                    face.landmarks[self.ALIGN_POINTS],
                    bean.landmarks[self.ALIGN_POINTS]
                )

                warped_bean_mask = self.warp_image(bean_mask, M, base_image.shape)
                combined_mask = numpy.max(
                    [self.get_face_mask(face), warped_bean_mask], axis = 0
                )

                warped_image = self.warp_image(bean.image, M, base_image.shape)
                cv2.imwrite('outputs/bean.image.jpg',bean.image)
                #コラの元画像、青色にする作業
                global parts
                frame = cv2.imread("outputs/bean.image.jpg")
                dets = detector(frame[:, :, ::-1])
                if len(dets) > 0:
                    parts = PREDICTOR(frame, dets[0]).parts()
                for i in parts:
                    cv2.circle(frame, (i.x, i.y), 3, (255, 0, 0), -1)
                #終わり
                cv2.imwrite('outputs/bean.blueface.jpg',frame)
                warped_corrected_image = self.correct_colors(base_image, warped_image, face.landmarks)
                base_image = base_image * (1.0 - combined_mask) + warped_corrected_image * combined_mask

            path, ext = os.path.splitext( os.path.basename(image_path) )
            cv2.imwrite('outputs/output_' + path + ext, base_image)
            if self.before_after is True:
                before_after = numpy.concatenate((original, base_image), axis = 1)
                cv2.imwrite('before_after/' + path + ext, before_after)

        def _draw_convex_hull(self, image, points, color):
            "指定したイメージの領域を塗りつぶす"
            points = cv2.convexHull(points)
            cv2.fillConvexPoly(image, points, color = color)
        #顔の向きのやつ↓
        def get_head_pose(self,shape):
            image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                    shape[39], shape[42], shape[45], shape[31], shape[35],
                                    shape[48], shape[54], shape[57], shape[8]])

            _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

            reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                                dist_coeffs)

            reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

            # calc euler angle
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            pose_mat = cv2.hconcat((rotation_mat, translation_vec))
            _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

            return reprojectdst, euler_angle

        def head_pose_estimation(self,image_path):
            
            frame = cv2.imread(image_path)
            face_rects = detector(frame, 0)

            if len(face_rects) > 0:
                shape = PREDICTOR(frame, face_rects[0])
                shape = face_utils.shape_to_np(shape)
                reprojectdst, euler_angle = self.get_head_pose(shape)

                global fx,fy,fz
                fx=euler_angle[0, 0]
                fy=euler_angle[1, 0]
                fz=euler_angle[2, 0]

        #顔の向き終わり↑
        def _load_beans(self,image_path):#顔のロードします
            "Mr. ビーンの画像をロードして、顔(特徴点など)を検出しておく"
            self.head_pose_estimation(image_path)
            self.beans = []
        #   print("ここで正面左右選択してくれえ")
            print(fy)
            print('a')
            
        def _get_bean_similar_to(self, face):#ここ
            "特徴点の差分距離が小さいMr.ビーンを返す"
            get_distances = numpy.vectorize(lambda bean: numpy.linalg.norm(face.landmarks - bean.landmarks))
            #self.beansに_load_beansにてロードした画像をすべてぶち込まれています。
            global DEGREE_OF_SIMILARITY
            
            if fy<=-17:
                distances = get_distances(rightface)
                """
                print(distances)
                print(distances.argmin()+1)
                print(distances[distances.argmin()])
                """
                DEGREE_OF_SIMILARITY=distances[distances.argmin()]
                return rightface[distances.argmin()]
            elif fy>=17:
                distances = get_distances(leftface)
                DEGREE_OF_SIMILARITY=distances[distances.argmin()]
                return leftface[distances.argmin()]
            else :
                distances = get_distances(frontface)
                DEGREE_OF_SIMILARITY=distances[distances.argmin()]
                return frontface[distances.argmin()]

    if __name__ == "__main__":
        global parts
        frame = cv2.imread("image.jpg")
        dets = detector(frame[:, :, ::-1])
        if len(dets) > 0:
            parts = PREDICTOR(frame, dets[0]).parts()
        for i in parts:
            cv2.circle(frame, (i.x, i.y), 3, (255, 0, 0), -1)
        cv2.imwrite('outputs/blueface.jpg',frame)
        be_bean = BeBean('image.jpg')
        be_bean.to_bean('image.jpg')
        print(len(rightface))
        cv2.destroyAllWindows()
    global root
    root = tk.Tk()
    root.geometry('1280x600')
    root.title('判定結果')  

    img={} 
    img[3] = Image.open('outputs/output_image.jpg')
    img[3] = img[3].resize((640,480))
  #  img[3].thumbnail((PICTURE_WIDTH, PICTURE_HEIGHT), Image.ANTIALIAS) #画像の大きさ最大値
    img[3] = ImageTk.PhotoImage(img[3])  # 表示するイメージを用意
    #3OK!!!!
    img[4] = Image.open('outputs/blueface.jpg')
    img[4].thumbnail((PICTURE_WIDTH, PICTURE_HEIGHT), Image.ANTIALIAS) #画像の大きさ最大値
    img[4] = ImageTk.PhotoImage(img[4])  # 表示するイメージを用意
    img[5] = Image.open('outputs/bean.blueface.jpg')
 #   img[5] = img[5].resize((PICTURE_WIDTH,PICTURE_HEIGHT))
    img[5].thumbnail((PICTURE_WIDTH, PICTURE_HEIGHT), Image.ANTIALIAS) #画像の大きさ最大値
    img[5] = ImageTk.PhotoImage(img[5])  # 表示するイメージを用意
    img[6] = Image.open('arrow.jpg')
    img[6].thumbnail((100, 100), Image.ANTIALIAS) #画像の大きさ最大値
    img[6] = ImageTk.PhotoImage(img[6])  # 表示するイメージを用意
    canvas={}
    #3 コラ画像出力
	################################
    canvas[3] = tk.Canvas(root,width=600,height=460)
    canvas[3].place(x=640,y=50)#画像の座標
    canvas[3].create_image(0,0,image=img[3],tag="illust",anchor=tk.NW)
    #4　青点々の顔画像出力
    canvas[4] = tk.Canvas(root,width=PICTURE_WIDTH,height=PICTURE_HEIGHT)
    canvas[4].place(x=20,y=50)#画像の座標
    canvas[4].create_image(0,0,image=img[4],tag="illust",anchor=tk.NW)
    #4　青点々の顔画像出力
    canvas[5] = tk.Canvas(root,width=PICTURE_WIDTH,height=PICTURE_HEIGHT)
    canvas[5].place(x=PICTURE_WIDTH+20+10,y=50)#画像の座標
    canvas[5].create_image(0,0,image=img[5],tag="illust",anchor=tk.NW)
	################################
    canvas[6] = tk.Canvas(root,width=100,height=100)
    canvas[6].place(x=535,y=250)#画像の座標
    canvas[6].create_image(0,0,image=img[6],tag="illust",anchor=tk.NW)
    ################################
    print(DEGREE_OF_SIMILARITY)
	# Buttonを設置してみる
    Button1 = tk.Button(text=u'消すボタン',)
    Button1.bind("<Button-1>", deleteEntry)        # ボタンが押されたときに実行される関数をバインドします
    Button1.place(x=960, y=550)
    var = tk.StringVar()
    var.set(round(DEGREE_OF_SIMILARITY))#round()関数、小数点切り捨てしてくれる
    label1 = tk.Label(text=u'あなたと右画像の有名人との類似度は', 
                    foreground='#ff0000',
                    background='#ffaacc',
                    font=("",15))
    label1.place(x=20,y=250)
    label2 = tk.Label(root,textvariable = var, 
                    fg="white", bg="black",font=("",20))
    label2.pack(side="top")
    label2.place(x=20, y=280)
####
    label3 = tk.Label(text=u'値が小さいほど似ています。', 
                    foreground='#000000',
                    font=("",10))
    label3.place(x=20,y=350)
####
    label4 = tk.Label(text=u'～500　あんまり似てないですね。がんばって', 
                    foreground='#000000',
                    background='#ffaacc',font=("",10),anchor="w")
    label4.place(x=20,y=370)
####
    label5 = tk.Label(text=u'499～300　まぁまぁにてるんじゃない？', 
                    foreground='#000000',
                    background='#ffaacc',font=("",10),anchor="w")
    label5.place(x=20,y=390)
####
    label6 = tk.Label(text=u'300～100　よく、似てますねぇ!', 
                    foreground='#000000',
                    background='#ffaacc',font=("",10),anchor="w")
    label6.place(x=20,y=410)
####
    label7 = tk.Label(text=u'99～10　　本人っぽい',foreground='#000000',
                    background='#ffaacc',font=("",10),anchor="w")
    label7.place(x=20,y=430)
####
    label8 = tk.Label(text=u'9～0　　　本人(サインください', foreground='#000000',
                    background='#ffaacc',font=("",10),anchor="w")
    label8.place(x=20,y=450)
####
    label8 = tk.Label(text=u'類似度について…あなたの顔のパーツの位置と有名人たちの顔のパーツの位置の差を計算します。\nその差が小さいほど類似度が高いです。',
    foreground='#000000',
    background='#ffaacc',font=("",10),anchor="w")
    label8.place(x=20,y=500)
####

    root.mainloop()	
#ChangeYourFaceFunction()関係終わり
#TakeFaceAndCheckFunction()関係
def TakeFaceAndCheckFunction():
    def takeFace(self):
        ret, image = c.read()
      #  cv2.imshow("Capture", image)
        cv2.imwrite("image.jpg", image)
        c.release()
        cv2.destroyAllWindows()
        root.destroy()

    def capStart():#camera-ON
        try:
            global w, h, img
            w, h= c.get(cv2.CAP_PROP_FRAME_WIDTH), c.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #    print('w:'+str(w)+'px+h:'+str(h)+'px')
        except:
            print("error-----")
            print(sys.exec_info()[0])
            print(sys.exec_info()[1])
            '''終了時の処理はここでは省略します。
            c.release()
            cv2.destroyAllWindows()'''

    def u():#update
        global img
        ret, frame =c.read()
        if ret:
            img=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            canvas.create_image(w/2,h/2,image=img)
        else:
            print("u-Fail")
        root.after(1,u)

#    print("TakeFaceAndCheckFunction")
    global root,c
    root=tk.Tk()
    root.title("camera")
    root.geometry("720x490")
    root.resizable(width=False, height=False)
    c=cv2.VideoCapture(0)

    # Buttonを設置してみる
    Button1 = tk.Button(text=u'撮影します')
    Button1.bind("<Button-1>", takeFace)        
    Button1.place(x=300, y=460)

    global canvas
    canvas=tk.Canvas(root, width=640, height=460, bg="white")
    canvas.pack()
    capStart()
    u()
    root.mainloop()
    CheckFunction()
#TakeFaceAndCheckFunction()関係終わり

###CheckFunction()関係
def CheckFunction():
    def Restartmain(self):
    #    print("写真撮り直し")
        root.destroy()
        TakeFaceAndCheckFunction()
#    print("CheckFunction")
    global root
    root=tk.Tk()
    root.geometry('720x510')
    root.resizable(width=False, height=False)
    root.title('IMG')
    img={}
    img[1] = Image.open('image.jpg')
    img[1].thumbnail((700, 500), Image.ANTIALIAS) #画像の大きさ最大値
    img[1] = ImageTk.PhotoImage(img[1])# 表示するイメージを用意
    ################################
    canvas={}
    canvas[1] = tk.Canvas(
		root, # 親要素をメインウィンドウに設定
		width=700,  # 幅を設定
		height=500 # 高さを設定
		#relief=tk.RIDGE  # 枠線を表示
		# 枠線の幅を設定
    )
	#PILでjpgを使用
    canvas[1].place(x=0,y=0)#画像の座標
    canvas[1].create_image(50,0,image=img[1],tag="illust",anchor=tk.NW)
	################################
	# Buttonを設置してみる
    Button1 = tk.Button(text=u'これでOK')
    Button1.bind("<Button-1>", deleteEntry)        # ボタンが押されたときに実行される関数をバインドします
    Button1.place(x=260, y=485)
    # Buttonを設置してみる
    # Buttonを設置してみる
    Button2 = tk.Button(text=u'撮り直す')
    Button2.bind("<Button-1>", Restartmain)        # ボタンが押されたときに実行される関数をバインドします
    Button2.place(x=320, y=485)
    root.mainloop()
#CheckFunction()関係終わり

def _load_pic():
    """
    class NoFaces(Exception):
        pass
    """
    class Face:
        def __init__(self, image, rect):
            self.image = image
            self.landmarks = numpy.matrix(
            [[p.x, p.y] for p in PREDICTOR(image, rect).parts()]
            )
    class BeBean:
        SCALE_FACTOR = 1
        FEATHER_AMOUNT = 11

        # オーバーレイする特徴点

        COLOR_CORRECT_BLUR_FRAC = 0.7
        def __init__(self, image_path, before_after = True):
            self.detector = dlib.get_frontal_face_detector()
            self._load_beans(image_path)
            self.before_after = before_after
        def load_faces_from_image(self, image_path):
            """
            画像パスから画像オブジェクトとその画像から抽出した特徴点を読み込む。
            ※ 画像内に顔が1つないし複数検出された場合も、返すので正確には「特徴点配列」の配列を返す
            """
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (image.shape[1] * self.SCALE_FACTOR,
                                    image.shape[0] * self.SCALE_FACTOR))
            rects = self.detector(image, 1)
            if len(rects) == 0:
                raise NoFaces
        #    else:出力がうるさいprint("Number of faces detected: {}".format(len(rects)))
            faces = [Face(image, rect) for rect in rects]
            return image, faces

        def _load_beans(self,image_path):#顔のロードします
            "Mr. ビーンの画像をロードして、顔(特徴点など)を検出しておく"
        #   print("ここで正面左右選択してくれえ")
            print('右')
            cnt=0
            global rightface,leftface,frontface
            rightface=[]
            leftface=[]
            frontface=[]
            for image_path in glob.glob(os.path.join('right', '*.jpg')):
                cnt=cnt+1
                print(cnt, end="")
                print(image_path)
                image, right_face = self.load_faces_from_image(image_path)
                rightface.append(right_face[0])
            
            print("左")
            for image_path in glob.glob(os.path.join('left', '*.jpg')):
                cnt=cnt+1
                print(cnt, end="")
                print(image_path)
                image, left_face = self.load_faces_from_image(image_path)
                leftface.append(left_face[0])
            
            print("正面")
            for image_path in glob.glob(os.path.join('front', '*.jpg')):
                cnt=cnt+1
                print(cnt, end="")
                print(image_path)
                image, front_face = self.load_faces_from_image(image_path)
                frontface.append(front_face[0])
            print(rightface)
    if __name__ == "__main__":
        be_bean = BeBean('image.jpg')

#ChangeYourFaceFunction()関係終わり

if __name__ == "__main__":
    cnt=0

    _load_pic()

    while True:
        TakeFaceAndCheckFunction()
        ChangeYourFaceFunction()#Esc,enterで脱出
        cnt=cnt+1
        if cnt == 100:
            break