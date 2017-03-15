# Machine Learning Engineer Nanodegree
# Reinforcement Learning
## Project: Train a Smartcab How to Drive

### Install

This project requires **Python 2.7** with the [pygame](https://www.pygame.org/wiki/GettingStarted
) library installed.

I included an environment.yaml to let you set up the environment using anaconda:
```conda env create -f environment.yaml```

Then activate the environment "smartcab".
For mac/linux:
```source activate smartcab```

For windows:
```activate smartcab```

Note that if installing pygame by yourself, if conda does not work, try uninstalling pygame, then installing using pip:
```pip install pygame```

From my experience, when installing pygame using the command:
```conda install -c https://conda.anaconda.org/quasiben pygame```

If I then run the smartcab, the visual simulation does not appear:
```python smartcab/agent.py```

To fix this, I uninstall pygame:
```conda remove pygame```

Then I install pygame using pip
```pip install pygame```

Then when I run the smartcab application, the visual simulation window appears:
```python smartcab/agent.py


### Code

Template code is provided in the `smartcab/agent.py` python file. Additional supporting python code can be found in `smartcab/enviroment.py`, `smartcab/planner.py`, and `smartcab/simulator.py`. Supporting images for the graphical user interface can be found in the `images` folder. While some code has already been implemented to get you started, you will need to implement additional functionality for the `LearningAgent` class in `agent.py` when requested to successfully complete the project. 

### Run

In a terminal or command window, navigate to the top-level project directory `smartcab/` (that contains this README) and run one of the following commands:

```python smartcab/agent.py```  
```python -m smartcab.agent```

This will run the `agent.py` file and execute your agent code.
