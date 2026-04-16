### Slide 1: Automated Seismic First-Break Picking
* **The Project Objective:** This project focuses on automating a foundational step in geophysics called "First-Break Picking." This involves pinpointing the exact millisecond wave energy first reaches a sensor after being sent into the ground.
* **The Impact of Accuracy:** If these initial time-markers are incorrect, the entire map of the subsurface becomes distorted. Precise "first breaks" are essential for calculating the speed of vibrations and creating clear images of what lies beneath the earth.
* **Visualization:** A high-quality title slide with a background showing a 3D subsurface map or a seismic waveform.

**Speech Section:**
Hello everyone. Today I’m presenting a project on "Seismic First-Break Picking." In simple terms, when we want to know what’s underground—like looking for minerals or oil—we send vibrations into the earth. Thousands of sensors then record when those vibrations bounce back. My project is about building an AI that can automatically and precisely mark the exact moment that first bit of energy hits each sensor. If we get this timing wrong, our final image of the underground will be blurry and useless, so precision is everything here.

---

### Slide 2: Project Scope and the HardPicks Dataset
* **The Workflow:** We developed an end-to-end machine learning system that handles everything from verifying raw data to training advanced neural networks. The goal was to take messy, real-world data and turn it into a reliable prediction tool.
* **The Data Source:** Our work is built on the "HardPicks" dataset, which consists of four major mining surveys from different locations: Brunswick, Halfmile, Lalor, and Sudbury. Each of these presents unique challenges in terms of noise and data quality.
* **Visualization:** A map or icons representing the four different survey locations (Brunswick, Halfmile, Lalor, Sudbury) to show the diversity of the data.

**Speech Section:**
This isn't just a simple AI model; it’s a full pipeline. We started with the "HardPicks" dataset, which is a famous collection of data from four different mining sites. Each site has its own quirks—some have a lot of "noise" (like background vibrations), while others have very clear signals. We built a system that verifies the raw files, cleans them up, and then tests several different types of AI brains to see which one can handle the variety of these four locations best.

---

### Slide 3: Presentation Roadmap
* **From Raw Data to Results:** We will walk through the scientific problem, the specific datasets used, and how we transformed raw 3D surveys into formats an AI can understand.
* **Deep Dive into AI Strategy:** We’ll also cover our "Exploratory Data Analysis," the different model architectures we tested, and a detailed look at the final results and performance metrics.
* **Visualization:** A clean, numbered list showing the Table of Contents (Project Summary, Scientific Problem, Datasets, etc.).

**Speech Section:**
To give you an idea of where we’re going: First, I’ll explain the scientific problem in more detail. Then, I’ll show you the specific data we used and how we transformed it from 3D physical space into 1D and 2D formats for the AI. Finally, we’ll look at the "Exploratory Data Analysis"—which is how we figured out what was wrong with the data—and then I’ll show you which AI models actually won the race.

---

### Slide 4: Why Automation is Necessary
* **The Human Bottleneck:** Historically, human experts had to manually look at these vibrations and click the starting point. While humans are good at this, modern surveys produce millions of traces, making manual work slow, expensive, and prone to mistakes.
* **Consistency in Noise:** Humans get tired and inconsistent, especially when the signal is weak or the background noise is loud. An AI can process millions of these "vibration lines" with the same level of attention and speed from start to finish.
* **Visualization:** A chart showing the exponential growth of data traces vs. the flat line of human processing speed.

**Speech Section:**
In the old days, a geologist would sit at a computer and click the start of every wave. But today, we have millions of these waves—called "traces"—in a single survey. It’s too much for any person to handle. Not only is it slow and expensive to pay humans to do this, but after a few hours, people get tired and start making mistakes, especially if the signal is hard to see. Our goal is to make this process instant and perfectly consistent.

---

### Slide 5: Defining the Target: Predicting a Physical Value
* **Measuring Time, Not Labels:** This is not a simple classification task where the AI says "Yes" or "No." Instead, the model must predict a continuous number—the exact arrival time in milliseconds on a physical axis.
* **The Waveform as Input:** The AI inspects a 1D waveform, which is a recording of vibrations over time. The output is a single floating-point number that geophysicists then use for deeper technical calculations.
* **Visualization:** A single "wiggly line" (seismic trace) with a red vertical line marking the "First Break" at a specific millisecond.

**Speech Section:**
To be clear about what the AI is doing: it’s not just looking for a "vibration." It’s looking for the *exact moment* that vibration begins. Most AI you hear about might classify an image as a "dog" or a "cat." Our AI has to give us a precise number, like "124.5 milliseconds." This isn't just a guess; it’s a physical measurement that other scientists will use later to build their maps, so the math has to be spot on.

---

### Slide 6: The Challenge of Data Harmonization
* **Incompatible "Languages":** The four surveys we used are fundamentally different; they use different timing grids, different recording windows, and different ways of scaling coordinates.
* **Engineering Compatibility:** We treat this as a "harmonization" problem. We don't assume the data is ready to use; we verify and re-engineer the raw files so that the AI can compare them on equal ground.
* **Visualization:** A diagram showing four different-sized blocks being "processed" into four identical, standardized blocks.

**Speech Section:**
Here is where it gets tricky: the four datasets I mentioned don't "talk" to each other. One site might record data every 1 millisecond, while another uses a completely different scale. Some have lots of labels for the AI to learn from, and others have almost none. A huge part of this project was "Harmonization"—essentially building a massive engineering pipeline that forces all these different file types into one common format so the AI doesn't get confused.

---

### Slide 7: Seeing vs. Hearing (1D vs. 2D)
* **Image vs. Waveform:** We explored two ways to represent data. One treats the data as a 1D sound wave, while the other treats a group of waves as a 2D image where patterns become visible.
* **The Missing Link (Offset):** The most important clue is the physical distance between the sound source and the sensor. We found that 1D models often "forget" this distance, which is a major limitation we had to address.
* **Visualization:** A side-by-side: A single wavy line (1D) vs. a 2D "shot gather" which looks like a grayscale image with a visible curved pattern.

**Speech Section:**
Finally, we had to decide how the AI should "see" the data. We tested two main ways. The first way is treating each recording like a single sound file (1D). The second way is looking at a whole group of recordings together, which creates a 2D image. In that image, you can actually see a visual curve forming. We discovered that if the AI only looks at the 1D sound, it misses the "big picture" of where the sensors are actually located, which is a major reason why some models fail where others succeed.