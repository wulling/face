from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
from flask import render_template
from cropImage import cropImage
from edit import edit
# import datetime
app = Flask(__name__,
static_folder='./dist',  #设置静态文件夹目录
template_folder = "./dist",
static_url_path="")

CORS(app, resources=r'/*')

@app.route('/')
def index():
    return render_template('index.html', name='index')

@app.route('/upload', methods=["POST"])
def calculate():
    # print(datetime.datetime.now())
    uid = uuid.uuid1()
    print(uid.hex)
    uuidhex = uid.hex;
    # 获取参数 颜色 尺寸 照片
    params = request.form
    color = params.get("color") # 尺寸
    size = params.get("size") # 尺寸 不用了前端进行剪切
    attribute = params.get("attribute")
    img_file = request.files.get('image') # 照片文件
    path = "uploadImg\\"
    suffix = ".jpg"
    orgImgName =path+uuidhex+"org"+suffix
    img_file.save(orgImgName)   # 保存文件
    
    edit_img = edit(orgImgName,attribute)
    editImgName =path+uuidhex+"edit"+suffix
    edit_img.save(editImgName)
   
    trimapImgName =path+uuidhex+"trimap"+suffix
    resultImgName = path+uuidhex+"result"+suffix
    # 成功result为处理后图片的路径 失败result为-1
    result = cropImage(editImgName, trimapImgName, resultImgName, color) #生成带有背景色的照片
    return result

def check_charset(file_path):
    import chardet
    with open(file_path, "rb") as f:
        data = f.read(4)
        charset = chardet.detect(data)['encoding']
    return charset

@app.route('/imagepreview', methods=["GET"])
def imagePreview():
    path = request.args.get("path")
    print(path)
    import base64
    img_stream = ''
    with open(path, 'rb') as img_f:
        img_stream = img_f.read()
        # img_stream = base64.b64encode(img_stream)
    return img_stream

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000)