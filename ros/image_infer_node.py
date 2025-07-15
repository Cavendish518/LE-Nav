#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray
import cv2
import numpy as np
import base64
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading
from uuid import uuid4

import os
import sys
from openai import OpenAI
import re

import torch
from model import CVAE # put model file in the same path during deployment

import dynamic_reconfigure.client
from ultralytics import YOLO


def imgmsg_to_cv2(img_msg):
    # in case CvBridge has conflict
    if img_msg.encoding not in ["bgr8", "rgb8"]:
        rospy.logerr("Unsupported encoding: {}. Only 'bgr8' and 'rgb8' are supported.".format(img_msg.encoding))
        return None

    dtype = np.dtype("uint8")
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')

    img_data = np.frombuffer(img_msg.data, dtype=dtype)

    expected_size = img_msg.height * img_msg.step
    if img_data.size != expected_size:
        rospy.logwarn("Image data size does not match expected dimensions. May result in reshape error.")

    image = img_data.reshape((img_msg.height, img_msg.step // 3, 3))

    if img_msg.step != img_msg.width * 3:
        image = image[:, :img_msg.width, :]

    if img_msg.is_bigendian == (sys.byteorder == 'little'): 
        image = image.byteswap().newbyteorder()

    if img_msg.encoding == "rgb8":
        image = image[:, :, ::-1]  # RGB → BGR
    return image


def load_model_for_inference(config_path='/your_path/config.yaml', model_path='/your_path/best_model_dwa.pth'):
    # load CVAE
    import yaml
    with open(config_path, 'r') as f:
        config_all = yaml.safe_load(f)
    config = config_all["param"]
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

    model = CVAE(
        input_dim=config['input_dim'],
        cond_dim=config['cond_dim'],
        feed_dim=config['feed_dim'],
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        mask=config['mask']
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, device, config

def generate_data(model, ctx, device,mask_indices,num_samples=1):
    # generate hyperparameters
    model.eval()
    with torch.no_grad():
        latent_dim = model.fc_mu.out_features
        z = torch.randn(num_samples, latent_dim)
        z = z.to(device)
        c = model.ctxencode(ctx, mask_indices=mask_indices)
        generated_data = model.decode(z, c)
    return generated_data

def re_normalize(x_tensor,planner = "TEB",device='cuda:0'):
    if planner == "TEB":
        norm_max = torch.tensor(np.array([0.4, 1.2, 0.4, 0.24, 0.24, 2.0, 2.0, 1.5, 50.0])).to(device)
        norm_min = torch.tensor(np.array([0.23, 0.59, 0.24, 0.15, 0.15, 0.99, 0.99, 0.99, 9.99])).to(device)
    elif planner == "DWA":
        norm_max = torch.tensor(np.array([0.4, 1.2, 1.8, 2.0, 5.0, 10.0, 32.0, 0.15, 0.4])).to(device)
        norm_min = torch.tensor(np.array([0.32, 0.5, 0.6, 1.2, 2.0, 1.0, 1.0, 0.05, 0.2])).to(device)
    return x_tensor * (norm_max - norm_min) + norm_min

def get_completion(
    messages: list[dict[str, str]],
    model: str = "model name" ,
    max_tokens=500,
    temperature=0,
    seed=123,
    tools=None,
    logprobs=None,
    top_logprobs=None
) -> str:
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    if tools:
        params["tools"] = tools
    client = OpenAI(
        base_url="your_url",
        api_key="your_api-key",
    )
    completion = client.chat.completions.create(**params)
    return completion

def MLLM(eg_url,img_url,test_model):
    INSTRUCTION_PROMPT = """
               ### As a visual analysis module of a robot, you will analyze the current environment and pedestrians based on the input image. Your analysis tasks are divided into two parts: pedestrians and the environment in the image.
               ## Please follow the thinking sequence below.
               # Question (1): Are there any pedestrians in the image? Please note that from the camera's perspective, some people in the image may only show their lower bodies, such as legs — do not overlook these pedestrians. If yes, the answer is 1; if no, the answer is 0.
               # Question (2): If the answer to Question (1) is 0, then the answer to Question (2) is 0. If the answer to Question (1) is 1, then the answer to Question (2) is the number of pedestrians in the image.
               # Question (3): If the answer to Question (1) is 0, then the answer to Question (3) is 0. Otherwise, are the pedestrians in the image walking toward you? Please judge whether each pedestrian in the image is facing forward. If any pedestrian’s face is facing forward, it means someone is walking toward you. If there are pedestrians walking toward you, the answer to Question (3) is 1; if not, the answer is 0.
               # Question (4): Analyze whether the environment (exclude pedestrian) is complex. The environment should be classified into one of the following three categories: complex (The answer will be 1), or simple (The answer will be 0), or not ordinary indoor scene (The answer will be 6). Follow the steps below: Identify all objects in the image other than pedestrians.
               If there is no door or furnitures and the scene is a spacious corridor or long hall, the environment is simple. If the scene contains any of the following objects: a door, a bed, a table, or a chair, it is considered to be complex.
           """

    EXAMPLE_PROMPT = """
               ## Below is an example analysis process of the first given image.
               # Question (1): There are pedestrians in the image, so the answer to Question (1) is 1.
               # Question (2): There are 2 pedestrians in the image. So the answer to Question (2) is 2.
               # Question (3): The pedestrian on the left is facing forward, indicating they are walking toward me. The pedestrian in the right is facing sideways, indicating they are not walking toward me. Overall analysis: there are pedestrians walking toward me, so the answer to Question (3) is 1.
               # Question (4): This is an indoor scene. Besides the road and walls, there are also doors. Therefore, the environment is complex, and the answer is 1.
               ## Based on the analysis process, the only answers you should provide are as below. You should also obey this format.
               1 2 1 1
            """
    Analysis_PROMPT = """  Now follow the given rules,instrutions and examples, analysis the second input image. Obey the answer format.
            """

    img_type = "image/jpeg"
    # indoor usage

    API_RESPONSE = get_completion(
        [
            {
                "role": "system",
                "content": INSTRUCTION_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img_type};base64,{eg_url}"
                        }
                    },
                ],
            },
            {
                "role": "system",
                "content": EXAMPLE_PROMPT,
            },
            {
                "role": "user",
                "content": Analysis_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img_type};base64,{img_url}"
                        }
                    },
                ],
            }
        ],
        model= test_model,
        temperature=0.3,
        logprobs=True,
        top_logprobs=1
    )


    reply = API_RESPONSE.choices[0].message.content
    if API_RESPONSE.choices[0].logprobs is not None:
        probs = [np.round(token.logprob,6) for token in API_RESPONSE.choices[0].logprobs.content]
    else:
        probs = None
    return reply, probs

def encode_image(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def detect_largest_person(image,near_ratio = 1/15):
    # visual model
    model = YOLO('/your_path/yolo11n.pt')
    results = model(image,classes=[0],device=0,conf=0.5,verbose = False)

    img_area = 640*480 # realsense

    boxes = results[0].boxes
    person_areas = []
    person_confs = []
    box = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    if len(boxes.cls) >= 1:
        for i in range(len(boxes.cls)):
            area = (box[i,2] - box[i,0]) * (box[i,3] - box[i,1])
            person_areas.append(area)
            person_confs.append(conf[i])


        max_idx = np.argmax(person_areas)
        max_area = person_areas[max_idx]
        max_conf = person_confs[max_idx]

        return max_area / img_area, np.round(float(max_conf),6)
    else:
        return 0.0, 0.0

def turn2array(reply, probs,near_score,conf):
    string_values = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', reply)]

    if len(string_values) != 4:
        raise ValueError(f"Expected 4 numbers, but got {len(string_values)} from reply: {reply}")

    if probs == None:
        selected_probs = string_values
    else:
        selected_probs = probs[:4]
    result = np.array(list(zip(string_values, selected_probs)))
    new_row = np.array([[near_score,conf]])
    result = np.vstack((result, new_row))
    return result

def encode_image_from_cv2(cv_image):
    _, buffer = cv2.imencode('.jpg', cv_image)
    return base64.b64encode(buffer.tobytes()).decode('utf-8')

class ImageInferNode:
    def __init__(self):
        rospy.init_node('image_infer_node', anonymous=True)
        self.client_global = dynamic_reconfigure.client.Client("/move_base/global_costmap/inflation_layer", timeout=5)
        self.planner = "DWA"
        # self.planner = "TEB"
        if self.planner  == "TEB":
            self.client_local = dynamic_reconfigure.client.Client('/move_base/TebLocalPlannerROS')
            self.conservative_param = np.array([0.24, 0.6, 0.25, 0.16, 0.16, 1.0, 1.0, 1.0, 10.0])
        elif self.planner == "DWA":
            self.client_local = dynamic_reconfigure.client.Client('/move_base/DWAPlannerROS')
            self.conservative_param = np.array([0.32, 0.6, 1.6, 1.2, 5.0, 1.0, 10.0, 0.05, 0.2])

        self.sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.pub = rospy.Publisher("/inference_output", Float32MultiArray, queue_size=10)
        self.time_pub = rospy.Publisher("/f_call_time", Float32, queue_size=10)
        self.input_pub = rospy.Publisher("/model_input", Float32MultiArray, queue_size=10)

        self.executor = ThreadPoolExecutor(max_workers=3)
        self.result_map = {}
        self.frame_queue = deque(maxlen=3)
        self.lock = threading.Lock()

        self.frame_interval = 2.0
        self.max_wait = 5.99
        self.last_frame_time = 0


    def image_callback(self, msg):
        now = time.time()
        if now - self.last_frame_time < self.frame_interval:
            return
        self.last_frame_time = now

        try:
            cv_image = imgmsg_to_cv2(msg)
        except Exception as e:
            rospy.logerr(f"cv_bridge error: {e}")
            return

        encoded = encode_image_from_cv2(cv_image)
        frame_id = str(uuid4())
        self.frame_queue.append(frame_id)

        self.executor.submit(self.call_f_and_store, frame_id, encoded,cv_image)
        threading.Thread(target=self.wait_and_infer, args=(frame_id,), daemon=True).start()

    def call_f_and_store(self, frame_id, encoded, cv_image):
        start = time.time()
        test_model = "your_model"

        # example
        example_path = "/your_path/000000.jpg"
        eg_url = encode_image(example_path)
        try:
            reply, probs = MLLM(eg_url, encoded, test_model)
            mlm_features = turn2array(reply, probs, 0, 0)[:4]
        except Exception as e:
            rospy.logerr(f"MLLM error: {e}")
            mlm_features = -1 * np.ones((4, 2))

        near_score, conf = detect_largest_person(cv_image)
        yolo_row = np.array([[near_score, conf]])

        result = np.vstack((mlm_features, yolo_row))
        duration = time.time() - start

        self.time_pub.publish(duration)

        with self.lock:
            self.result_map[frame_id] = (result, time.time())

    def update_inflation_async(self,new_radius):
        def task():
            try:
                self.client_global.update_configuration({"inflation_radius": new_radius})
            except Exception as e:
                rospy.logerr("Failed to update inflation_radius: %s", e)
        threading.Thread(target=task).start()

    def update_local_config_async(self, params_local):
        def task():
            try:
                self.client_local.update_configuration(params_local)
            except Exception as e:
                rospy.logerr("Failed to update local planner config: %s", e)

        threading.Thread(target=task).start()

    def wait_and_infer(self, frame_id):
        timeout = 2.6 # change here if you have better network
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                if frame_id in self.result_map:
                    break
            time.sleep(0.05)

        features = []
        with self.lock:
            for fid in list(self.frame_queue)[-3:]:
                entry = self.result_map.get(fid, None)
                if entry and time.time() - entry[1] <= self.max_wait and entry[0] is not None:
                    features.append(entry[0])
                else:
                    features.append(-1 * np.ones((5, 2)))
        model_input = np.stack(features, axis=-1)
        input_msg = Float32MultiArray(data=model_input.flatten().tolist())
        self.input_pub.publish(input_msg)
        result = self.run_model(model_input)
        self.publish_result(result)

    def run_model(self, ctx_input):
        ctx_tensor = torch.from_numpy(ctx_input).float()
        model, device, config = load_model_for_inference()
        ctx_tensor = ctx_tensor.unsqueeze(0).to(device)
        pack_drop = (ctx_tensor[:, :4, :, :] == -1).all(dim=2).all(dim=1)
        pack_drop_num = torch.sum(pack_drop)
        print("pack_drop_num:", pack_drop_num.item())
        if ctx_tensor.shape[3] != 3:
            return self.conservative_param
        elif pack_drop_num == 3:
            return self.conservative_param
        else:
            with torch.no_grad():
                generated = generate_data(model, ctx_tensor, device,mask_indices=pack_drop, num_samples=1)
                generated = re_normalize(generated.squeeze(0),planner= self.planner)
            conservative_choice = 0.5 # for deployment if your network is fine change it to 0.0
            unconf_score = conservative_choice*pack_drop_num / pack_drop.size(1)
            output = np.round(generated.cpu().numpy(),3) * (1-unconf_score.cpu().numpy()) + unconf_score.cpu().numpy() * self.conservative_param
            output = np.round(output,3)
            return output

    def publish_result(self, result):
        msg = Float32MultiArray(data=result.tolist())
        self.pub.publish(msg)
        self.update_inflation_async(result[0])

        if self.planner == "TEB":
            params_local = {
            "max_vel_x": result[1],
            "max_vel_theta": result[2],
            "acc_lim_x": result[3],
            "acc_lim_theta": result[4],
            "weight_max_vel_x": result[5],
            "weight_acc_lim_x": result[6],
            "weight_acc_lim_theta": result[7],
            "weight_optimaltime": result[8]
            }
        elif self.planner == "DWA":
            params_local = {
            'max_vel_trans': result[1],
            "max_vel_x": result[1],
            "max_vel_theta": result[2],
            "acc_lim_x": result[3],
            "acc_lim_theta": result[4],
            "path_distance_bias": result[5],
            "goal_distance_bias": result[6],
            "occdist_scale": result[7],
            "forward_point_distance": result[8]
            }
        self.update_local_config_async(params_local)


if __name__ == "__main__":
    node = ImageInferNode()
    rospy.spin()
