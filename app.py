import streamlit as st
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb
from PIL import Image
import os
import tensorflow as tf
import numpy as np
import PIL.Image
import time
import functools
from io import BytesIO
import tensorflow_hub as hub
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def process_image(content_path,style_path):
    content_image = load_img(content_path)
    style_image = load_img(style_path)      
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    return tensor_to_image(stylized_image)
    

def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))

def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 90px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(5, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        "Made in ",
        image('https://avatars3.githubusercontent.com/u/45109972?s=400&v=4',
              width=px(25), height=px(25)),
        " with ❤️ by ",
        link("https://www.linkedin.com/in/kishanmishra1/", "@Kishanmishra1"),
        " ",
        " ~ ",
        image('https://cdn4.iconfinder.com/data/icons/iconsimple-logotypes/512/github-1024.png',width=px(18), height=px(18)),
        " : ",
        link("https://github.com/KishanMishra1/CreateArt", "CreateArt")
        
    ]
    layout(*myargs)
reduce_header_height_style = """
    <style>
        div.block-container {padding-top:1rem;}
    </style>
"""




def main():
    
    st.set_page_config(
        page_title="CreateArt",
        page_icon="🏞",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title('CreateArt 🏞')
    footer()
    
    st.markdown(reduce_header_height_style, unsafe_allow_html=True)
    st.header('A Neural Algorithm of Artistic Style.\n')
    
    col1, col2 ,col3 = st.columns(3,gap='large')
    spin=False
    with col1:
        st.subheader('Upload the Content Image ')
        content_file = st.file_uploader("Content Image")
        if content_file:
            img=Image.open(content_file)
            st.image(img,width=500) 
            img.save("./content_images/content.png")

    with col2:
        st.subheader('Upload the Style Image ')
        style_file = st.file_uploader("Style Image")
        if style_file:
            img2=Image.open(style_file)
            img2.save("./style_images/style.png")
            st.image(img2,width=500) 
            
    with col3:
        st.subheader('Resulted Artistic Image ')
        if content_file and style_file:
            with st.spinner('Creating Art....'):
                content_path='./content_images/content.png'
                style_path='./style_images/style.png'
                img3=process_image(content_path=content_path,style_path=style_path)
                st.image(img3,width=500)
                img3.save('result.png')
            response=st.download_button(label='Download Image',
                        data= open('result.png', 'rb').read(),
                        file_name='artistic_image.png',
                        mime='image/png')
                    
                    
                
        
if __name__=='__main__': 
    main()

    


