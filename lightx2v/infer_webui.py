# coding=utf-8

import argparse

import torch.distributed as dist
from loguru import logger

from lightx2v.common.ops import *
from lightx2v.models.runners.cogvideox.cogvidex_runner import CogvideoxRunner  # noqa: F401
from lightx2v.models.runners.graph_runner import GraphRunner
from lightx2v.models.runners.hunyuan.hunyuan_runner import HunyuanRunner  # noqa: F401
from lightx2v.models.runners.qwen_image.qwen_image_runner import QwenImageRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_audio_runner import Wan22AudioRunner, WanAudioRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_causvid_runner import WanCausVidRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_distill_runner import WanDistillRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_runner import Wan22MoeRunner, WanRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_skyreels_v2_df_runner import WanSkyreelsV2DFRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_vace_runner import WanVaceRunner  # noqa: F401
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import ProfilingContext
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import print_config, set_config, set_parallel_config
from lightx2v.utils.utils import seed_all


def init_runner(config):
    seed_all(config.seed)

    if CHECK_ENABLE_GRAPH_MODE():
        default_runner = RUNNER_REGISTER[config.model_cls](config)
        default_runner.init_modules()
        runner = GraphRunner(default_runner)
    else:
        runner = RUNNER_REGISTER[config.model_cls](config)
        runner.init_modules()
    return runner

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_cls",
    type=str,
    required=True,
    choices=[
        "wan2.1",
        "hunyuan",
        "wan2.1_distill",
        "wan2.1_causvid",
        "wan2.1_skyreels_v2_df",
        "wan2.1_vace",
        "cogvideox",
        "seko_talk",
        "wan2.2_moe",
        "wan2.2",
        "wan2.2_moe_audio",
        "wan2.2_audio",
        "wan2.2_moe_distill",
        "qwen_image",
    ],
    default="wan2.1",
)

parser.add_argument("--task", type=str, choices=["t2v", "i2v", "t2i", "i2i", "flf2v", "vace"], default="t2v")
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--config_json", type=str, required=True)
parser.add_argument("--use_prompt_enhancer", action="store_true")

parser.add_argument("--prompt", type=str, default="", help="The input prompt for text-to-video generation")
parser.add_argument("--negative_prompt", type=str, default="")

parser.add_argument("--image_path", type=str, default="", help="The path to input image file for image-to-video (i2v) task")
parser.add_argument("--last_frame_path", type=str, default="", help="The path to last frame file for first-last-frame-to-video (flf2v) task")
parser.add_argument("--audio_path", type=str, default="", help="The path to input audio file for audio-to-video (a2v) task")

parser.add_argument(
    "--src_ref_images",
    type=str,
    default=None,
    help="The file list of the source reference images. Separated by ','. Default None.",
)
parser.add_argument(
    "--src_video",
    type=str,
    default=None,
    help="The file of the source video. Default None.",
)
parser.add_argument(
    "--src_mask",
    type=str,
    default=None,
    help="The file of the source mask. Default None.",
)

parser.add_argument("--save_video_path", type=str, default="./output_lightx2v.mp4", help="The path to save video path/file")
args = parser.parse_args()

# set config
config = set_config(args)

if config.parallel:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    set_parallel_config(config)

print_config(config)

# with ProfilingContext("Total Cost"):
runner = init_runner(config)


# coding=utf-8

import json,traceback,os,warnings
import shutil
import time
import logging,logging,logging.handlers
import traceback
now_dir = os.getcwd()
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
logging.getLogger("multipart.multipart").setLevel(logging.ERROR)
logging.getLogger("python_multipart.multipart").setLevel(logging.ERROR)
logging.getLogger("split_lang.split.splitter").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
formatter = logging.Formatter('%(asctime)s\tline:%(lineno)d\t\t%(message)s',"%Y-%m-%d %H:%M:%S")
logger1 = logging.getLogger(__name__+"flask_gpu_server")
logger1.setLevel(level=logging.INFO)
handler1 = logging.handlers.TimedRotatingFileHandler("%s/logs/webui.log"%now_dir, when='D', interval=1, backupCount=365)
handler1.setLevel(logging.INFO)
handler1.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger1.addHandler(handler1)
logger1.addHandler(console)
logger1_info=logger1.info

import gradio,gradio as gr
import requests,threading
from time import time as ttime
str2size={
    "720P":(1280,720),
    "540P":(960,544),
    "360P":(640,360),
}
from PIL import Image
def generate(prompt, img, seed, nf, motion, size_str, shift, step):
    try:
        tt = int(ttime())
        if type(img) == type(None):
            gr.Error("首帧图必须输入")
            raise KeyError
        save_file = "%s/output/%s-%s-%s-%s-%s-%s.mp4" % (now_dir,seed, nf, shift, step, size_str, tt)

        prompt += " aesthetic score: 5.5. motion score: %s. " % motion
        prompt += "There is no text in the video."

        # with open("%s/prompt_logs/%s.txt" % (now_dir,tt), "w") as ffff:
        #     ffff.write(prompt)
        # image_path="%s/prompt_logs/%s.png"% (now_dir,tt)
        # Image.fromarray(img).save(image_path)

        # shutil.copy(image_path,image_path)

        seed = seed if seed >= 0 else random.randint(1, 999999)
        w,h=str2size[size_str]
        if size_str=="720P"and nf>7:
            gradio.Warning("720P目前最多支持到生成7s，超过则以7s生成。")
            nf=7
        more_config = {
            "prompt": prompt,
            # "image_path": image_path,
            "image_path": img,
            "save_video_path": save_file,
            "seed": seed,
            "target_video_length": int(nf) * 16 + 1,
            "sample_shift": shift,
            "infer_steps": step,
            "target_width": w,
            "target_height": h,
        }
        config.update(more_config)
        logger.info(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")
        runner.config = config
        runner.run_pipeline()
        video_update = gr.update(visible=True, value=save_file)
        seed_update = gr.update(visible=True, value=seed)
        return save_file, video_update, seed_update
    except:
        info=traceback.format_exc()
        logger1_info(info)


with gr.Blocks() as demo:
    gr.Markdown("""
           <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
               AniSora-Bilibili动画视频生成模型
           </div>
           """)
    with gr.Row():
        with gr.Column():
            with gr.Accordion("I2V", open=True):  # 544x960
                # image_path = gr.Image(label="输入图像")
                image_path = gr.Image(label="输入图像")
            prompt = gr.Textbox(label="Prompt (Less than 200 Words)", placeholder="Enter your prompt here", lines=5)
            with gr.Row():
                nf = gr.Slider(label="秒数", minimum=3, maximum=8, step=0.5, value=5)
                motion = gr.Slider(label="运动幅度，越大越高", minimum=0.5, maximum=10, step=0.1, value=1.3)
                shift = gr.Slider(label="shift", minimum=1, maximum=17, step=1, value=5, visible=False)
                step = gr.Slider(label="step", minimum=3, maximum=12, step=1, value=8, visible=False)
                size_str = gr.Radio(label="分辨率", value="540P", choices=["360P", "540P", "720P"])

                seed = gr.Number(
                    label="种子，请输入正整数，-1为随机", value=233
                )

            generate_button = gr.Button("Generate Video")

        with gr.Column():
            video_output = gr.Video(label="Generated Video")
            with gr.Row():
                download_video_button = gr.File(label="? Download Video", visible=False)
                seed_text = gr.Number(label="Seed Used for Video Generation", visible=False)

    generate_button.click(
        generate,
        inputs=[prompt, image_path, seed, nf, motion, size_str, shift, step],
        outputs=[video_output, download_video_button, seed_text],
    )
    # with gr.Row(visible=True):
    #     gr.Examples([
    #         [
    #             "In the video, a white-haired girl dances as the camera zooms in. She sings while rotating her right hand toward the lens, fingers spread wide.", "assets/mmd1.jpg", 233, 5, 1.3, 5, 8, "assets/233-5-原版-1747397279.mp4"
    #         ],
    #         [
    #             "The scene depicts an exploding rock, erupting in blinding light as shattered fragments blast outward in all directions.", "assets/big2.jpg", 233, 5, 1.3, 5, 8, "assets/233-5-加速版-1747402956.mp4"
    #         ],
    #     ],
    #         inputs=[prompt, image_path, seed, nf, motion, shift, step, video_output], )

demo.queue(max_size=4).launch(
    server_name="0.0.0.0",
    inbrowser=True,
    share=True,
    server_port=12345,
    # quiet=True,
)

# from fastapi import FastAPI
# import uvicorn
# app = FastAPI()
# @app.get('/v2/health/ready')
# def health():
#     return ""
# demo.queue(max_size=15)
# app = gr.mount_gradio_app(app, demo, path="/api/adhoc/ttv/demo")
# uvicorn.run(app, host="0.0.0.0", port=26780)  #

# Clean up distributed process group
if dist.is_initialized():
    dist.destroy_process_group()
    logger.info("Distributed process group cleaned up")

