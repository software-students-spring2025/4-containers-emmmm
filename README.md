![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)

# Containerized App Exercise
---

## Team Members
- **Eli Sun**: [IDislikeName](https://github.com/IDislikeName)
- **Jasmine Zhang**: [Jasminezhang666666](https://github.com/Jasminezhang666666)
- **Yifan Zhang**: [YifanZZZZZZ](https://github.com/YifanZZZZZZ)
- **Shuyuan Yang**: [shuyuanyyy](https://github.com/shuyuanyyy)

---

## Project Overview

Our project is containerized web application that allows users to record audio in the browser, and detects the emotional content analyzing by machine learning models.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/software-students-spring2025/4-containers-emmmm.git
cd 4-containers-emmmm
```


### 2. Start the System with Docker

```bash
docker-compose build --no-cache
docker-compose up  
```

### 3. Visit the Web through Browser

```bash
http://127.0.0.1:6000
```

---

## References
The pre-trained model in machine learning client part is from Hugging Face https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP, which is an audio emotion recognition model. 
