#!/usr/bin/env python3

import os
import sys
import dlib
import cv2
import subprocess
import os
import requests
import time
import hashlib


# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import onnxruntime
import tensorflow
import roop.globals
import roop.metadata
import roop.ui as ui
#from roop.predictor import predict_image, predict_video
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    program.add_argument('-s', '--source', help='select an source image', dest='source_path')
    program.add_argument('-t', '--target', help='select an target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('--frame-processor', help='frame processors (choices: face_swapper, face_enhancer, ...)', dest='frame_processor', default=['face_swapper'], nargs='+')
    program.add_argument('--keep-fps', help='keep target fps', dest='keep_fps', action='store_true')
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true')
    program.add_argument('--skip-audio', help='skip target audio', dest='skip_audio', action='store_true')
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true')
    program.add_argument('--reference-face-position', help='position of the reference face', dest='reference_face_position', type=int, default=0)
    program.add_argument('--reference-frame-number', help='number of the reference frame', dest='reference_frame_number', type=int, default=0)
    program.add_argument('--similar-face-distance', help='face distance used for recognition', dest='similar_face_distance', type=float, default=0.85)
    program.add_argument('--temp-frame-format', help='image format used for frame extraction', dest='temp_frame_format', default='png', choices=['jpg', 'png'])
    program.add_argument('--temp-frame-quality', help='image quality used for frame extraction', dest='temp_frame_quality', type=int, default=0, choices=range(101), metavar='[0-100]')
    program.add_argument('--output-video-encoder', help='encoder used for the output video', dest='output_video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc'])
    program.add_argument('--output-video-quality', help='quality used for the output video', dest='output_video_quality', type=int, default=35, choices=range(101), metavar='[0-100]')
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int)
    program.add_argument('--execution-provider', help='available execution provider (choices: cpu, ...)', dest='execution_provider', default=['cpu'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('-v', '--version', action='version', version=f'{roop.metadata.name} {roop.metadata.version}')

    args = program.parse_args()

    roop.globals.source_path = args.source_path
    roop.globals.target_path = args.target_path
    roop.globals.output_path = normalize_output_path(roop.globals.source_path, roop.globals.target_path, args.output_path)
    roop.globals.headless = roop.globals.source_path is not None and roop.globals.target_path is not None and roop.globals.output_path is not None
    roop.globals.frame_processors = args.frame_processor
    roop.globals.keep_fps = args.keep_fps
    roop.globals.keep_frames = args.keep_frames
    roop.globals.skip_audio = args.skip_audio
    roop.globals.many_faces = args.many_faces
    roop.globals.reference_face_position = args.reference_face_position
    roop.globals.reference_frame_number = args.reference_frame_number
    roop.globals.similar_face_distance = args.similar_face_distance
    roop.globals.temp_frame_format = args.temp_frame_format
    roop.globals.temp_frame_quality = args.temp_frame_quality
    roop.globals.output_video_encoder = args.output_video_encoder
    roop.globals.output_video_quality = args.output_video_quality
    roop.globals.max_memory = args.max_memory
    roop.globals.execution_providers = decode_execution_providers(args.execution_provider)
    roop.globals.execution_threads = args.execution_threads


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        return 8
    return 1


def limit_resources() -> None:
    # prevent tensorflow memory leak
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_virtual_device_configuration(gpu, [
            tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
        ])
    # limit memory usage
    if roop.globals.max_memory:
        memory = roop.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = roop.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed.')
        return False
    return True


def update_status(message: str, scope: str = 'ROOP.CORE') -> None:
    print(f'[{scope}] {message}')
    if not roop.globals.headless:
        ui.update_status(message)


def start() -> None:

    print('测试', roop.globals.frame_processors)
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_start():
            return
    # process image to image
    if has_image_extension(roop.globals.target_path):
 #       if predict_image(roop.globals.target_path):
  #          destroy()
        shutil.copy2(roop.globals.target_path, roop.globals.output_path)
        # process frame
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            
      
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_image(roop.globals.source_path, roop.globals.output_path, roop.globals.output_path)
            frame_processor.post_process()
        # validate image
        if is_image(roop.globals.target_path):
            update_status('Processing to image succeed!')
        else:
            update_status('Processing to image failed!')
        return
    # process image to videos
#    if predict_video(roop.globals.target_path):
#        destroy()
    update_status('Creating temporary resources...')
    create_temp(roop.globals.target_path)
    # extract frames

    fps = detect_fps(roop.globals.target_path)
    update_status(f'Extracting frames with {fps} FPS...')
    if fps > 30:
        fps = 30;
    extract_frames(roop.globals.target_path, fps)
    
    # process frame
    temp_frame_paths = get_temp_frame_paths(roop.globals.target_path)

    #总帧数传上去
    print('发送总帧数:', len(temp_frame_paths))
 #   addLog(0, 2, 'frame', 0, len(temp_frame_paths))

    if temp_frame_paths:
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            print('frame_processor.NAME:', frame_processor.NAME, frame_processor.NAME == 'ROOP.FACE-ENHANCER')
            if frame_processor.NAME == 'ROOP.FACE-ENHANCER' and  not int(taskData.get('is_enhancement', 0)):
                continue;
            frame_processor.process_video(roop.globals.source_path, temp_frame_paths)
            frame_processor.post_process()
    else:
        update_status('Frames not found...')
        return
    # create video
    if roop.globals.keep_fps:
        fps = detect_fps(roop.globals.target_path)
        update_status(f'Creating video with {fps} FPS...')
        create_video(roop.globals.target_path, fps)
    else:
        update_status('Creating video with 30 FPS...')
        create_video(roop.globals.target_path)
    # handle audio
    if roop.globals.skip_audio:
        move_temp(roop.globals.target_path, roop.globals.output_path)
        update_status('Skipping audio...')
    else:
        if roop.globals.keep_fps:
            update_status('Restoring audio...')
        else:
            update_status('Restoring audio might cause issues as fps are not kept...')
        restore_audio(roop.globals.target_path, roop.globals.output_path)
    # clean temp
    update_status('Cleaning temporary resources...')
    clean_temp(roop.globals.target_path)
    # validate video
    if is_video(roop.globals.target_path):
        update_status('Processing to video succeed!')
    else:
        update_status('Processing to video failed!')


def destroy() -> None:
    if roop.globals.target_path:
        clean_temp(roop.globals.target_path)


def proc_video(input_video_filename, face_path, out_video_filename):
#    shutil.copy2(input_video_filename, out_video_filename)
#    return
    '''
    # 加载 dlib 的人脸检测器
    face_detector = dlib.get_frontal_face_detector()

    cap = cv2.VideoCapture(input_video_filename)

    min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0
    frame_skip_interval = 30
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 使用 dlib 进行人脸检测
            faces = face_detector(gray)
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)
        frame_count += 1

    cap.release()

    # 计算最小矩形区域的位置和尺寸
    rect_x = min_x
    rect_y = min_y
    rect_width = max_x - min_x
    rect_height = max_y - min_y

    print("矩形区域的位置和尺寸:")
    print("左上角坐标 (x, y):", rect_x, rect_y)
    print("宽度:", rect_width)
    print("高度:", rect_height)

    ffmpeg_command = [
            'ffmpeg',
            '-i', input_video_filename,
            '-vf', f'crop={rect_width}:{rect_height}:{rect_x}:{rect_y}',
            '-y',  # 强制覆盖
            'media_sub.mp4'
            ]
    subprocess.run(ffmpeg_command)
    '''
    
    #roop.globals.execution_providers = ['CUDAExecutionProvider']
  #  roop.globals.execution_providers = ['CPUExecutionProvider']
    roop.globals.execution_threads = 8
 #   roop.globals.frame_processors = ['face_swapper', 'face_enhancer']
    roop.globals.headless = True
    roop.globals.keep_fps = True
    roop.globals.keep_frames = True
    roop.globals.log_level = 'error'
    roop.globals.many_faces = False
    roop.globals.max_memory = None
    roop.globals.output_path = 'media_out.mp4'
    roop.globals.output_video_encoder = 'libx264'
    roop.globals.output_video_quality = 35
    roop.globals.reference_face_position = 0
    roop.globals.reference_frame_number = 0
    roop.globals.similar_face_distance = 1.5
    roop.globals.skip_audio = False
    roop.globals.source_path = face_path
    roop.globals.target_path = input_video_filename#'media_sub.mp4'
    roop.globals.temp_frame_format = 'jpg'
    roop.globals.temp_frame_quality = 1

    start()
    
'''
    ffmpeg_command = [
            'ffmpeg',
            '-i', input_video_filename,
            '-i', 'media_sub_out.mp4',
            '-filter_complex', f'[0:v][1:v]overlay=x={rect_x}:y={rect_y}:enable=\'between(t,0,10)\'',
            '-c:a', 'copy',
            '-y',  # 强制覆盖
            'media_out.mp4'
            ]

    subprocess.run(ffmpeg_command)
'''
def proc_image(input_image_filename, face_path, out_image_filename):
  #  roop.globals.execution_providers = ['CUDAExecutionProvider']
 #   roop.globals.execution_providers = ['CPUExecutionProvider']
    roop.globals.execution_threads = 8
   # roop.globals.frame_processors = ['face_swapper', 'face_enhancer']
    roop.globals.headless = True
    roop.globals.keep_fps = True
    roop.globals.keep_frames = True
    roop.globals.log_level = 'error'
    roop.globals.many_faces = False
    roop.globals.max_memory = None
    roop.globals.output_path = out_image_filename
    roop.globals.output_video_encoder = 'libx264'
    roop.globals.output_video_quality = 35
    roop.globals.reference_face_position = 0
    roop.globals.reference_frame_number = 0
    roop.globals.similar_face_distance = 1.5
    roop.globals.skip_audio = False
    roop.globals.source_path = face_path
    roop.globals.target_path = input_image_filename
    roop.globals.temp_frame_format = 'jpg'
    roop.globals.temp_frame_quality = 1
    start()
def calculate_md5(input_string):
    md5_hash = hashlib.md5(input_string.encode()).hexdigest()
    return md5_hash
def upload_file(url, file_path):
    chunk_size = 1024 * 1024  # 1MB
    total_chunks = -(-os.path.getsize(file_path) // chunk_size)  # 總分片數，無條件取整
    current_chunk = 0
    data = {'name': '', 'link': ''}
    fileSize = os.path.getsize(file_path)
    upFileName = str(time.time()) + str(fileSize) + file_path
    with open(file_path, 'rb') as f:
        while current_chunk < total_chunks:
            start = current_chunk * chunk_size
            end = min(start + chunk_size, os.path.getsize(file_path))
            chunk = f.read(end - start)
            
            files = {
                'file': (os.path.basename(file_path), chunk),
                'chunkNumber': (None, str(current_chunk + 1)),
                'totalChunks': (None, str(total_chunks)),
                'fileSize': (None, fileSize),
                'fileName': (None, upFileName)
            }
            
            response = requests.post(url, files=files)
            print(response.text)
            res_json = response.json()
            if res_json['status'] != 'success':
                print(f'Upload error: Chunk {current_chunk + 1} / {total_chunks}')
                return data
            
            current_chunk += 1
            data = res_json
        
        data['name'] = os.path.basename(file_path)
        data['hash'] = 'hash_placeholder'  # 請替換為計算哈希值的方法
        print(f'文件大小：{os.path.getsize(file_path)}  {data["size"]}')
        # 調用捕獲視頻畫面並上傳縮略圖的函數，並將返回的數據更新到 data 對象中
        # tres = capture_video_frame_and_upload(file_path)
        # data['thumb'] = tres['thumb']
        
        print('Upload success!')
        return data



def download_file(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
            print(f"File '{filename}' downloaded and saved successfully.")
    else:
        print(f"Error downloading file from {url}")

import requests

def upload_image(upload_url, image_path):
    files = {'file': (image_path, open(image_path, 'rb'), 'image/jpeg')}
    try:
        response = requests.post(upload_url, files=files)
        response_json = response.json()  # 尝试解析JSON响应
        
        if response_json['status'] == 'success':
            print('Upload successful')
            return response_json
        else:
            print('Upload failed')
            print(response_json)
    except Exception as e:
        print('Error:', str(e))


def generate_video_thumbnail(video_path, thumbnail_path, max_size=512):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    if ret:
        # 获取视频帧的宽和高
        frame_height, frame_width, _ = frame.shape
        
        # 调整缩略图的大小
        if frame_width > max_size or frame_height > max_size:
            if frame_width > frame_height:
                new_width = max_size
                new_height = int(frame_height * (max_size / frame_width))
            else:
                new_height = max_size
                new_width = int(frame_width * (max_size / frame_height))
            frame = cv2.resize(frame, (new_width, new_height))
        
        # 保存缩略图
        cv2.imwrite(thumbnail_path, frame)
    
    cap.release()

def calculate_md5(input_string):
    md5_hash = hashlib.md5(input_string.encode()).hexdigest()
    return md5_hash


def callApi(name, data):
    try:
        #TODO 做簽名認證
        response = requests.post('http://192.3.153.102:3000/api/' + name, data)
        response_json = response.json()
        if response.status_code == 200:
            print('Request successful')
            return response_json
        else:
            print('Request failed')
            print(response_json.get('message', 'Unknown error'))
    except Exception as e:
        print('Error:', str(e))

def addLog(finish, state, log, process, total_frame = 0):
    callApi("workerUpdateTask", {'task_id':taskData['_id'], 'total_frame':total_frame,'finish':finish, 'state':state, 'log':log, 'process':process})
def gif2mp4(gif, mp4):
    ffmpeg_command = [
        'ffmpeg',
        '-i', gif,
        '-c:v', 'libx264',  # 使用H.264编码器
        '-pix_fmt', 'yuv420p',  # 设置像素格式，通常需要
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # 将高度和宽度调整为2的倍数
        '-y',  # 强制覆盖
        mp4
    ]
    subprocess.run(ffmpeg_command)
  
    
def mp42gif(input_mp4_filename, output_gif_filename):
    ffmpeg_command = [
        'ffmpeg',
        '-y',  # 强制覆盖
        '-i', input_mp4_filename,
        '-vf', 'fps=10,scale=320:-1:flags=lanczos',  # 设置帧速率和尺寸等参数
        output_gif_filename
    ]
    subprocess.run(ffmpeg_command)

def delete_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"文件 '{file_path}' 已被删除。")
        else:
            print(f"文件 '{file_path}' 不存在，无需删除。")
   
def work():
    global taskData
    data = callApi("workerGetTask", {})
    if data["code"] != 0:
        print("Error: Code is not 0.")
        time.sleep(3)
        return
    try:
        if os.path.exists(roop.globals.target_path):
            clean_temp(roop.globals.target_path)
        delete_files(['face.png','media.gif','media.png','media.mp4','media_out.gif','media_out.mp4','media_out.jpg'])
        print(f"temp have been removed.")
    except Exception as e:
        print(f"Error deleting directory: {e}")

    taskData = data['data']

    media_file_url = ''
    face_file_url = ''
    if data['data']['media'] == False:
        return;
    if data['data']['face'] == False:
        return;


    # roop.globals.frame_processors = ['face_swapper', 'face_enhancer']
    print('parse_args result:', roop.globals)
    if data['data']['is_enhancement']:
        roop.globals.frame_processors = ['face_swapper', 'face_enhancer']
    print('parse_args result:', roop.globals.frame_processors)
        
    try:
        media_file_url = data['data']['media']['file_url']
        face_file_url = data['data']['face']['file_url']
    except Exception as e:
        print(f"error get media_file_url: {e} {data}")
    
    media_filename = "media" + os.path.splitext(media_file_url)[1]
    face_filename = "face" + os.path.splitext(face_file_url)[1]
        
    download_file(media_file_url, media_filename)
    download_file(face_file_url, face_filename)

    extName = os.path.splitext(media_file_url)[1].lower()
        
    if media_filename.lower().endswith(('.mp4', '.m4v', '.mkv', '.avi', '.mov', '.webm', '.mpeg', '.mpg', '.wmv', '.flv', '.asf', '.3gp', '.3g2', '.ogg', '.vob', '.rmvb', '.ts', '.m2ts', '.divx', '.xvid', '.h264', '.avc', '.hevc', '.vp9', '.avchd')):
        
        out_file_path = 'media_out.mp4'
        proc_video(media_filename, face_filename, out_file_path)
        thumb_file_path = 'thumb_media.jpg'
        generate_video_thumbnail(out_file_path, thumb_file_path)
        if not os.path.exists(out_file_path):
            print(f"找不到文件 {out_file_path}")
            addLog(1, -1, 'Processing failed', 99)
            return
        upload_video_res = upload_file('https://fakeface.io/upload.php?m=media', out_file_path)
        upload_image_res = upload_image('https://fakeface.io/upload.php?m=thumb', thumb_file_path)
        print('Upload result:', upload_video_res, upload_image_res)
        api_res = callApi("wokerAddMedia", {'user_id':data['data']['user_id'], 'media_id':data['data']['finish_media_id'], 'file_url':upload_video_res['link'], 'thumb_url':upload_image_res['thumb'], 'file_hash':upload_video_res['size']})
        print('Api result:', api_res)
        addLog(1, 3, 'finish', 100)
        return

    if media_filename.lower().endswith(('.gif')):
        out_file_path = 'media_out.mp4'
        print('文件后缀：', extName)
        gif2mp4('media.gif', 'media.mp4')
        proc_video(media_filename, face_filename, out_file_path)
        thumb_file_path = 'thumb_media.jpg'
        generate_video_thumbnail(out_file_path, thumb_file_path)
        mp42gif('media_out.mp4', 'media_out.gif')
        out_file_path = 'media_out.gif'
        if not os.path.exists(out_file_path):
            print(f"找不到文件 {out_file_path}")
            addLog(1, -1, 'Processing failed', 99)
            return
        upload_video_res = upload_file('https://fakeface.io/upload.php?m=media', out_file_path)
        upload_image_res = upload_image('https://fakeface.io/upload.php?m=thumb', thumb_file_path)
        print('Upload result:', upload_video_res, upload_image_res)
        api_res = callApi("wokerAddMedia", {'user_id':data['data']['user_id'], 'media_id':data['data']['finish_media_id'], 'file_url':upload_video_res['link'], 'thumb_url':upload_image_res['thumb'], 'file_hash':upload_video_res['size']})
        print('Api result:', api_res)
        addLog(1, 3, 'finish', 100)
        return
    if media_filename.lower().endswith(('.jpg')):
        out_file_path = 'media_out.jpg'
        real_out_file_path = 'media_out.jpg'
        proc_image(media_filename, face_filename, out_file_path)

        if not os.path.exists(out_file_path):
            print(f"找不到文件 {out_file_path}")
            addLog(1, -1, 'Processing failed', 99)
            return
     #   addLog(0, 2, 'finish quickly', 99)
        upload_res = upload_image('https://fakeface.io/upload.php?m=png', out_file_path)
        
        print('Upload result:', upload_res)
        api_res = callApi("wokerAddMedia", {'user_id':data['data']['user_id'], 'media_id':data['data']['finish_media_id'], 'file_url':upload_res['link'], 'thumb_url':upload_res['thumb'], 'file_hash':'121212'})
        print('Api result:', api_res)
        addLog(1, 3, 'finish', 100)
        return
    addLog(1, 3, 'wrong file format', 100)

def run() -> None:
    parse_args()
       
    if not pre_check():
        return
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_check():
            return
    limit_resources()
    while True:
        work()

