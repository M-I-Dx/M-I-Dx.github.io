---
layout: homepage
---

## About Me

Driven machine learning engineer with a passion for advancing AI in diverse fields. Blending technical skills with creative solutions to drive technological breakthroughs.

## Education

### University of British Columbia, Vancouver, BC, Canada
**Master of Data Science in Computational Linguistics**  
*September 2023 - June 2024*

- **Key Courses:**
  - **Advanced Natural Language Processing:** Focused on transformer models, sequence-to-sequence tasks, and domain-specific language models.
  - **Machine Learning and Optimization:** Covered supervised and unsupervised learning, optimization algorithms, and their applications in large-scale data problems.
  - **Computational Linguistics:** Explored syntax, semantics, and discourse analysis using computational methods.
  - **Data Visualization and Interpretation:** Techniques for visualizing complex data, focusing on interactivity and clarity.

- **Relevant Projects:**
  - **Biomedical Lay Summarization using Large Language Models:**
    - **Objective:** Developed an innovative approach to make biomedical research more accessible to non-expert audiences by generating simplified summaries of complex scientific articles.
    - **Methodology:**
      - **Data Collection:** Curated a diverse set of biomedical research articles from various journals and repositories, focusing on those with complex technical content.
      - **Model Selection:** Employed domain-specific large language models (LLMs) like **BioMistral7b** to ensure the generated summaries maintained high fidelity to the original scientific content while being understandable to laypersons.
      - **Techniques Used:**
        - **Retrieval-Augmented Generation (RAG):** Integrated RAG to enhance the model’s ability to pull relevant information from external documents, improving the accuracy and relevance of the summaries.
        - **Representation Engineering:** Applied advanced representation techniques to better capture the nuances of biomedical terminology and ensure the model’s outputs were both accurate and easy to understand.
        - **Control Vectors:** Implemented control vectors to dynamically adjust the model's output, optimizing for readability and factual accuracy. This ensured that the summaries retained the key messages of the original articles while simplifying the language.
      - **Fine-Tuning:** Fine-tuned the model on a specifically designed corpus that included layman-friendly biomedical summaries, which were carefully annotated for clarity and simplicity.
    - **Outcomes:** 
      - Successfully generated summaries that were both accurate and easily understandable, bridging the gap between complex biomedical research and the general public.
      - The model demonstrated a significant improvement in producing readable and accurate summaries compared to baseline models, with particular effectiveness in simplifying dense technical content without losing essential information.
      - This project has potential applications in healthcare communications, patient education, and public health awareness campaigns, where simplifying complex information is crucial.

  - **Development of an Annotated Corpus for Problem-Solving Explanations:**
    - **Objective:** The primary goal of this project was to enhance the problem-solving capabilities of language models by developing an annotated corpus based on the AI2 Reasoning Challenge (ARC) dataset. The project aimed to generate natural language explanations for solving each problem in the dataset, thereby enabling the fine-tuning of a language model that could follow a structured "Tree of Thought" prompting approach.
    - **Methodology:**
      - **Dataset Selection:** Started with the **AI2 Reasoning Challenge (ARC)** dataset, which contains challenging multiple-choice science questions that require reasoning and problem-solving skills beyond simple fact retrieval.
      - **Annotation Process:**
        - Developed a detailed annotation scheme that involved creating step-by-step natural language explanations for each problem in the ARC dataset.
        - These explanations were structured according to a "Tree of Thought" framework, breaking down the problem-solving process into a series of logical, sequential steps.
      - **Model Fine-Tuning:**
        - Used the annotated corpus to fine-tune a pre-trained language model, focusing on improving its ability to follow complex reasoning paths.
        - The fine-tuning process emphasized the model's understanding of the problem-solving steps, enabling it to generate coherent and logical explanations when faced with similar problems.
      - **Tree of Thought Prompting:**
        - Implemented the "Tree of Thought" prompting technique to guide the model's reasoning process, ensuring that each step in the problem-solving sequence was clearly articulated and logically consistent.
    - **Outcomes:**
      - **Enhanced Problem-Solving Ability:** The fine-tuned model demonstrated a significant improvement in its ability to solve complex problems from the ARC dataset, effectively following the structured reasoning process laid out in the annotated corpus.
      - **Improved Explanation Generation:** The model was able to generate clear, step-by-step explanations for each problem, making its reasoning process transparent and easier to understand.
      - **Potential Applications:** This approach can be applied to other domains where complex reasoning and decision-making are required, such as legal reasoning, medical diagnosis, and educational tools that teach critical thinking skills.
      - **Research Contribution:** The project contributed to the field by showing how structured annotations and prompting techniques can be used to enhance the reasoning capabilities of language models, providing a framework for future research in explainable AI.

- **Extracurricular Activities:**
  - **AI Reading Club Leader and Presenter:**
    - Led an **AI Reading Club** at the University of British Columbia, where I organized weekly meetings focused on discussing recent advancements in artificial intelligence and machine learning.
    - Presented and led discussions on key papers, including topics such as transformer models, explainable AI, and reinforcement learning, fostering a collaborative learning environment.
    - Facilitated knowledge sharing among peers, helping members to deepen their understanding of complex AI topics and stay updated with the latest research trends.

### Thapar Institute of Engineering and Technology, Patiala, Punjab, India
**Bachelor of Technology in Computer Engineering**  
*August 2019 - July 2023*

- **Key Courses:**
  - **Data Structures and Algorithms:** In-depth study of algorithms, their complexities, and data structure optimization.
  - **Artificial Intelligence and Machine Learning:** Introduction to AI, covering fundamental algorithms and machine learning models.
  - **Database Management Systems:** Design and implementation of relational databases, focusing on SQL and NoSQL databases.
  - **Software Engineering:** Software development lifecycle, agile methodologies, and project management.

- **Relevant Projects:**
  - **Explainable AI for Synthetic Aperture Radar (SAR) Image Classification:**
    - **Objective:** The primary goal of this project was to develop a deep learning model capable of classifying synthetic aperture radar (SAR) images with a focus on enhancing the explainability of the model's predictions. This project aimed to provide clear, interpretable results that could be easily understood by domain experts and end-users alike.
    - **Methodology:**
      - **Data Collection:** Utilized publicly available SAR image datasets, focusing on military and environmental monitoring applications where precise target classification is crucial.
      - **Model Development:**
        - Designed and implemented a convolutional neural network (CNN) tailored for SAR image classification, optimizing the model for both accuracy and speed.
        - Integrated explainable AI techniques such as Local Interpretable Model-agnostic Explanations (LIME) and SHapley Additive exPlanations (SHAP) to provide insights into the decision-making process of the model.
        - Conducted extensive hyperparameter tuning and model validation to ensure robustness and reliability of the predictions.
      - **Explainability Enhancement:**
        - Employed LIME and SHAP to generate visual explanations that highlight the parts of the SAR images that most influenced the model’s decisions.
        - Developed a user interface that allows users to interact with the model and visualize the explanations, making it easier for non-technical stakeholders to understand the model's predictions.
    - **Outcomes:**
      - **Improved Model Interpretability:** Successfully enhanced the interpretability of the SAR image classification model, allowing users to understand and trust the model’s decisions.
      - **High Classification Accuracy:** Achieved high accuracy rates in SAR image classification, demonstrating the effectiveness of the CNN architecture combined with explainability techniques.
      - **Practical Applications:** This project has potential applications in military target recognition, environmental monitoring, and disaster management, where understanding the "why" behind AI predictions is critical.
      - **Research Contribution:** The project contributed to the field of explainable AI by demonstrating how traditional black-box models can be made more transparent and trustworthy, providing a framework for future research in interpretable machine learning.

  - **COVID-19 CT Scan Analysis Using Generative Adversarial Networks (GANs):**
    - **Objective:** Developed a deep learning model to enhance the detection accuracy of COVID-19 from CT scan images by generating synthetic CT scans using GANs, focusing on improving the dataset's diversity and the model's ability to generalize.
    - **Methodology:**
      - **Data Collection:** Gathered a comprehensive dataset of COVID-19 CT scans from open-source medical image repositories, ensuring a diverse range of cases and conditions.
      - **Model Development:**
        - Implemented a GAN architecture to generate high-quality synthetic CT scan images, addressing the challenge of limited annotated medical datasets.
        - Trained a CNN on both real and synthetic CT scans to improve the model’s ability to detect COVID-19, with a focus on enhancing its sensitivity and specificity.
      - **Evaluation and Validation:**
        - Conducted extensive evaluations comparing the performance of the model trained with and without synthetic data, demonstrating significant improvements in detection accuracy.
        - Applied standard medical image analysis metrics such as Dice coefficient, sensitivity, and specificity to validate the model’s performance.
    - **Outcomes:**
      - **Enhanced Detection Accuracy:** The model achieved superior accuracy in detecting COVID-19 from CT scans, particularly in challenging cases where data was previously limited.
      - **Data Augmentation Success:** Successfully demonstrated the use of GANs for data augmentation in medical imaging, improving the generalization capabilities of the model.
      - **Research Contribution:** Contributed to the ongoing research in using GANs for medical data augmentation, providing valuable insights into how synthetic data can be leveraged to improve deep learning models in healthcare.
      - **Publication Potential:** The outcomes of this project hold potential for publication in medical AI and machine learning journals, especially in the context of pandemic response and healthcare innovation.

- **Extracurricular Activities:**
  - Core mentor of the **Developer Student Club**, organizing coding competitions, projects and workshops.
  - Active participant in Hackathons and Open Source projects.

## Skills

- **Programming Languages:**
  - **Proficient:** Python, R, MATLAB
  - **Familiar:** JavaScript, Bash, SQL

- **Machine Learning and Data Science:**
  - **Core Libraries:** Pandas, NumPy, SciPy
  - **Machine Learning Frameworks:** Scikit-Learn, XGBoost, LightGBM, CatBoost
  - **Deep Learning Frameworks:** PyTorch, PyTorch Lightning, TensorFlow, Keras, Hugging Face
  - **Natural Language Processing:** NLTK, spaCy, Transformers, BERT, GPT, RNNs, LSTMs
  - **Optimization:** Hyperparameter tuning with Optuna, Bayesian Optimization, Grid Search, Random Search
  - **Data Processing:** Numba, Dask
  - **Databases:** PostgreSQL, MongoDB

- **Deep Learning:**
  - **Architectures:** Transformers, CNNs, RNNs, LSTMs, GANs, Autoencoders
  - **Specialized Models:** Large Language Models (LLM), BERT, GPT, VAE, DCGAN, BioMistral, 
Llama
  - **Techniques:** Transfer Learning, Fine-Tuning, Model Pruning, Quantization, Knowledge Distillation

- **Data Visualization and Interpretation:**
  - **Visualization Libraries:** Altair, Plotly, Plotly-Dash, Matplotlib, Seaborn, ggplot2
  - **Interactive Dashboards:** Streamlit, Dash
  - **Reporting Tools:** Datapane, Tableau

- **DevOps and MLOps:**
  - **Version Control:** Git, GitHub, GitLab, GitHub Actions
  - **Containerization and Orchestration:** Docker, Kubeflow, Kubernetes
  - **Experiment Tracking and Model Management:** Weights & Biases (Wandb)

- **Problem Solving and Analytical Skills:**
  - **Algorithm Design:** Dynamic Programming, Graph Algorithms, Greedy Algorithms, Divide and Conquer
  - **Data Structures:** Arrays, Linked Lists, Trees, Graphs, Hash Tables, Heaps
  - **Complexity Analysis:** Big-O notation, Space-Time Trade-offs
  - **Debugging and Optimization:** Profiling, Memory Optimization, Code Refactoring

- **Research and Development:**
  - **Academic Research:** Literature Review, Hypothesis Testing, Statistical Analysis
  - **Paper Writing:** LaTeX, Academic Writing, Research Methodologies
  - **Presentation:** Public Speaking, Data Storytelling, Visualization for Communication

- **Soft Skills:**
  - **Leadership:** Team Management, Project Leadership, Mentorship
  - **Communication:** Technical Writing, Documentation, Cross-functional Collaboration
  - **Creativity:** Innovative Problem Solving, Design Thinking, Ideation Techniques

## Work Experience

### Betterdata (Remote, Singapore)
**Machine Learning Intern**  
*February 2022 - July 2023*

- Spearheaded research in **tabular synthetic data generation**, developing and testing multiple generative model architectures to ensure data diversity, fidelity, and privacy.
- Integrated differential privacy into generative models, enhancing data protection standards while maintaining model performance.
- Collaborated with cross-functional teams to adapt generative models for deployment in both cloud and on-premises environments.
- Designed and implemented an evaluation system to assess the relevance, accuracy, and privacy of the generated synthetic data.

- **Technologies Used:** Utilized **PyTorch** for model development, **Numba** for parallel processing optimization, and **Scikit-learn** for statistical analysis.
- **Tools and Platforms:** Employed **Docker** for containerization, **AWS** for cloud-based deployment, and **TensorFlow** for initial model prototyping.

- **Quantitative Impact:** Reduced data generation time by 40% through optimized parallel processing and improved model deployment efficiency by 30% across cloud platforms.
- **Performance Metrics:** Achieved a 15% increase in data fidelity and a 20% enhancement in privacy-preserving capabilities of generative models, as measured by internal benchmarks.

### SpaceML (Remote, USA)
**AI Researcher**  
*February 2021 - February 2022*

- Developed an AI system in collaboration with NASA's IMPACT team, focusing on **Self-Supervised and Active Learning** techniques to efficiently process petabyte-scale satellite imagery datasets.
- Spearheaded the creation of a scalable **Active Learning pipeline**, reducing manual image labeling time from 7000 hours to just 52 minutes.
- Contributed to the development of a high-quality open-source labeling package for NASA's Phenomenon portal, facilitating more efficient data labeling.
- Led research initiatives that resulted in securing a NASA Science Mission Directorate grant, placing in the top 5 out of 79 proposals.

- **Technologies Used:** Applied **PyTorch** and **TensorFlow** for model development, with **AWS** and **GCP** for cloud scalability.
- **Tools and Platforms:** Leveraged **GitHub** for open-source collaboration, **Jupyter Notebooks** for experiment tracking, and **Kubernetes** for scaling Active Learning pipelines.

- **Quantitative Impact:** Reduced image labeling costs by over 85% through the implementation of automated Active Learning pipelines.
- **Performance Metrics:** Improved model accuracy by 25% in identifying relevant satellite imagery and achieved a 90% reduction in labeling time for large datasets.

### Thapar Institute of Engineering and Technology, Patiala, Punjab, India
**Undergraduate Student Researcher**  
*January 2020 - July 2022*

- Conducted research on **convolutional neural networks (CNNs)**, exploring the impact of hyperparameter tuning and architectural modifications on model performance and interpretability.
- Investigated the application of **generative adversarial networks (GANs)** to generate synthetic COVID-19 CT and MRI scans, enhancing the robustness of detection systems.
- Developed machine learning models for forecasting social media traction, focusing on feature engineering and model selection for optimal predictive performance.
- Collaborated with faculty and research teams to publish findings in peer-reviewed journals, contributing to the academic understanding of explainable AI and synthetic data generation.

- **Technologies Used:** Used **TensorFlow** and **Keras** for CNN and GAN development, with **OpenCV** for image processing.
- **Tools and Platforms:** Employed **SQL** and **Pandas** for data manipulation and analysis, and **Matplotlib** for visualizing model outputs.

- **Quantitative Impact:** Improved model interpretability by 30% through the integration of explainability techniques like **LIME** and **SHAP**.
- **Performance Metrics:** Enhanced the accuracy of COVID-19 detection models by 20% using synthetic data, as validated by cross-validation techniques.

### Bournemouth University (Remote, UK)
**Research Assistant**  
*June 2021 - August 2021*

- Developed a real-time camera-based **grape leaf disease diagnosis iOS app**, utilizing deep learning to provide accurate disease detection and remedy suggestions.
- Researched and optimized **Convolutional Neural Network (CNN)** architectures to balance accuracy with performance constraints on resource-limited mobile devices.
- Implemented model **quantization** and **pruning** techniques to ensure that the deep learning model ran efficiently on iPhone 8 (2017) and above.
- Collaborated with interdisciplinary teams to refine the app's user interface and improve the overall user experience for farmers and agricultural professionals.

- **Technologies Used:** Developed the app using **Swift** and **Core ML**, with **TensorFlow Lite** for model deployment.
- **Tools and Platforms:** Utilized **Xcode** for iOS app development, **Git** for version control, and **JIRA** for project management.

- **Quantitative Impact:** Improved app performance by 50% through model optimization, reducing inference time to under 500ms per image.
- **Performance Metrics:** Achieved over 90% accuracy in disease detection across multiple test environments, as validated by field trials and user feedback.

{% include_relative _includes/publications.md %}

