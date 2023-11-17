# import streamlit as st

# st.title("First streamlit app - Learning")
# st.header("header")
# st.subheader("sub-header")
# st.text("Hey, this is how text will look") 
# st.markdown("# This is a markdown") ## using # is important
# st.markdown("## This is a markdown")
# st.markdown("### This is a markdown")
# st.success("Green color text")
# st.error("Red color text")
# st.info("Information")
# st.warning("This is a warning")
# st.exception("NameError('name pd is not defined')")

# # import pandas
# # st.help(pandas)
# st.help(range)
# st.write("This will display text in writing waala format")
# st.write(range(10))

# from PIL import Image
# img=Image.open("t.jpg")
# # st.image(img)
# st.image(img,width=400,caption="Simple Image Caption")


# if st.checkbox("Show / Hide"):
#     st.text('Showing or hiding widget')


# # Radio buttons
# status = st.radio("What is your status",("Single","Committed"))

# if status=='Single':
#     st.success("You are single")
# else:
#     st.warning("You are committed")




# # https://docs.streamlit.io/library/get-started/main-concepts
# import pandas as pd
# df = pd.DataFrame({
#   'first column': [1, 2, 3, 4],
#   'second column': [10, 20, 30, 40]
# })

# df # Any time that Streamlit sees a variable or a literal value on its own line, it automatically writes that to your app using st.write(). !!! Not Recommended !!!  




# # Select Box
# occupation = st.selectbox("Select you occupation: ",["SDE",  "Frontend Developer",  "Mobile Developer",  "Backend Developer",  "Full Stack Engineer",  "QnA Associate",  "HR Intern",  "Data Scientist",  "UX/UI Designer",  "Product Manager",  "DevOps Engineer",  "Machine Learning Engineer",  "Data Analyst",  "Quality Assurance Engineer",  "Technical Writer",  "Systems Architect",  "Network Engineer",  "Cloud Solutions Architect", "UI/UX Researcher", "Content Strategist", "Cybersecurity Analyst", "Business Analyst", "Sales Engineer", "Embedded Systems Engineer", "Game Developer", "IT Support Specialist", "Video Editor", "Social Media Manager"]) 
# st.write("You've choosen",occupation)

# # Now to choose/select multiple options
# company = st.multiselect("Now u can choose multipe options : ", [ "Tata Consultancy Services (TCS)", "Infosys", "Wipro", "HCL Technologies", "Tech Mahindra", "Cognizant", "Capgemini", "Mindtree", "L&T Infotech", "Zoho Corporation", "ThoughtWorks", "Dell Technologies", "Adobe India", "NVIDIA India", "Microsoft India", "Amazon India", "Flipkart", "Paytm", "Swiggy", "Ola Cabs"])
# st.write("Here are your choices:",company)
# st.write("You've selected",len(company),"choices")


# # SLider
# sliderRange = st.slider("Easy Slider",1,10)

# # Buttons
# st.button("Simple button") 
# if(st.button("About")):
#     st.text("Its a lovely day nani")
# if(st.button("Submit")):
#     st.text("Successful submission")


# # Text-input
# first_name = st.text_input("Enter your First Name","John Doe") 
# if st.button("Submit",key="1"):
#     result = first_name.title()
#     st.success(result)


# # Text-Area
# textAreaExample = st.text_area("Enter your message","")
# if st.button("Submit",key="2"):
#     result = textAreaExample.title()
#     st.success(result)


# # Date and time Input
# import datetime
# today = st.date_input("Today is",datetime.datetime.now())
# st.time_input("Now time is",datetime.time())



# # Displaying output in json format
# st.text("Displaying output in json format")
# st.json({'foo':'bar','fu':'ba'})


# # Outputting raw code
# st.code('''import torch
# from PIL import Image
# from transformers import CLIPProcessor, CLIPModel
# ''')

# # ANother way of Displaying raw code with multiple lines
# with st.echo():
#     import torch
#     import pandas as pf 
#     from PIL import Image
#     from transformers import CLIPProcessor, CLIPModel
#     pd.DataFrame()


# # # Progress bar

# # import time
# # # Create an empty slot for the progress bar
# # bar = st.empty()
# # # Define the total number of iterations
# # total_iterations = 10
# # # Update the progress bar in a loop
# # for i in range(total_iterations):
# #     # Calculate the progress percentage
# #     progress = (i + 1) / total_iterations
# #     # Update the progress bar value
# #     bar.progress(progress)
# #     # Add a small delay (e.g., 0.5 seconds) to visualize the progress
# #     time.sleep(0.5)



# # # Spinnner
# # with st.spinner("Loading..."):
# #     time.sleep(5)
# # st.success("Loaded")    





# st.header('Happy TTimezz balloons')
# st.balloons()
# st.snow()
# st.toast("This is a toaster example")



# # Sidebar
# st.sidebar.header("About Section")
# st.sidebar.text("Sidebar of the project")
# st.sidebar.subheader("sub-header")
# st.sidebar.text("Hey, this is how text will look") 
# st.sidebar.markdown("# This is a markdown") ## using # is important
# st.sidebar.markdown("## This is a markdown")
# st.sidebar.markdown("### This is a markdown")
# st.sidebar.success("Green color text")
# st.sidebar.error("Red color text")
# st.sidebar.info("Information")
# st.sidebar.warning("This is a warning")
# st.sidebar.selectbox("Algorithms",[ "Linear Regression", "Logistic Regression", "Decision Trees", "Random Forests", "Support Vector Machines (SVM)", "K-Nearest Neighbors (KNN)", "Naive Bayes", "Principal Component Analysis (PCA)", "Gradient Boosting", "Neural Networks", "Clustering (e.g., K-Means)", "Natural Language Processing (NLP)", "Reinforcement Learning", "Time Series Analysis", "Deep Learning (DL)"])









# Actual App Building
import streamlit as st
from PIL import Image 
from io import BytesIO
from transformers import ViltProcessor,ViltForQuestionAnswering



st.set_page_config(page_title="A2 ML MAJOR PROJ",page_icon="random",layout="wide",)


processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")


def result(image,userQuestion):
    try:
        img = Image.open(BytesIO(image)).convert("RGB")
        encoding = processor(img,userQuestion,return_tensors="pt")

        res = model(**encoding)
        logits = res.logits
        idx = logits.argmax(-1).item()
        answer = model.config.id2label[idx]
        return answer

    except Exception as e:
        return str(e)        



st.title("Image Driven Interactive System using Vision Learning and NLP")
# st.balloons()

col1, col2 = st.columns(2)

with col1:
    st.markdown(''':red[Image Upload Section]''')
    upload=st.file_uploader(label="Upload an image to get started!", type=['png','jpg','jpeg'])
    if upload is not None:
        st.image(upload,use_column_width=True)

with col2:
    st.markdown(''':green[Output Section]''')
    question = st.text_input(label="Your Question",placeholder="Ask me anything related to the image..")

    if upload and question is not None:
        if st.button("Go!"):
            # st.text("Frontend done")
            image = Image.open(upload)
            imageByteArray = BytesIO()
            image.save(imageByteArray,format="jpeg")
            imageInBytes = imageByteArray.getvalue()

            answer = result(imageInBytes,question)
            st.info("The questionn was: "+ question)
            st.success("The answer is: " + answer)



