# HandsIn: Bridging Language Barriers in Healthcare Through AI-Driven ASL Training

Welcome to HandsIn, an innovative platform designed to empower healthcare providers with essential American Sign Language (ASL) skills. Our AI-driven solution aims to bridge communication gaps between healthcare providers and Deaf and Hard-of-Hearing patients, improving the quality of care and fostering a more inclusive environment.

## Story

The inspiration behind HandsIn comes from a personal experience. In 2019, my mom was diagnosed with hearing loss. As her hearing declined, everyday activities became challenging, often requiring a third person to facilitate communication. Even after receiving hearing aids, the experience left a lasting impact. Inspired by her advocacy for people with disabilities, we set out to address a critical issue in the Deaf and Hard-of-Hearing community—the need for healthcare providers to understand ASL rather than just translate it.

## The Problem

We discovered that in many crucial scenarios—such as emergency care, elderly homes, and classrooms—ASL is often overlooked or minimized in favor of English-based communication. This can lead to isolation and hinder the growth of individuals whose native language is ASL. HandsIn seeks to address these challenges by providing an interactive platform for learning and practicing ASL in real-world contexts.

## Our Solution

HandsIn offers an AI-powered avatar that adapts to user input and facilitates spontaneous conversations in ASL. This dynamic training platform is ideal for healthcare providers, students, and emergency responders who need to be prepared for situations where language and cultural understanding are vital.

## How We Built It

Our project is built on a three-step plan: 

1. Classify signs from a user.
2. Translate to ASL gloss.
3. Generate a video of an avatar signing back to the user.

### Technologies Used

- **Frontend:** Streamlit, a Python web framework, is utilized for all frontend development.
- **Backend:**
  - **Library Learning Paths:** These paths are built using a 3-input Artificial Neural Network real-time object detector, scaled to detect 24 letters of the alphabet.
  - **AI Avatar:** This comprises a chatbot that translates English input into ASL Gloss, processed through a Gloss-to-Video database, and presented via a 3D Avatar Generator.

### Deployment

HandsIn is deployed using Streamlit’s Cloud Service, and our domain was purchased from GoDaddy.

## Challenges

Our major challenge was finding a suitable program to convert videos to an avatar. When faced with API access limitations, we developed a Python script to automate the login and video upload/download process.

## Accomplishments

We successfully integrated LLMs, built our own Neural Network for hand movement classification, and created a workaround for API limitations. Our hackathon experience also taught us valuable lessons in community building and collaboration.

## Future Plans

We aim to expand HandsIn by developing industry-specific lessons (healthcare, education, etc.) and enhancing our model to recognize more signs. Additionally, we plan to include educational content on ASL's history and culture to enrich the experience for both Hearing and Deaf communities.

## Built With

- Python
- Streamlit
- Firebase
- Gemini AI API
- OpenCV
- PyTorch
- Scikit-learn

## Try it out

Experience HandsIn for yourself by visiting [hand-detection.streamlit.app](https://hand-detection.streamlit.app).

## Contributors

- **Grace Janusz**: Project initiation, backend support, AI model maker.
- **Defne Meric Erdogan**: Frontend integration and Firebase/Streamlit.
- **Elena Loucks**: Front End UI/UX Design.
- **Nusret Efe Ucer**: Scraping the Web, and training the models.

## Join the Conversation

Leave feedback, and follow our journey on Devpost.

---

We look forward to growing HandsIn and continuing to make a meaningful impact in the healthcare and communication sectors. Thank you for your interest and support!
