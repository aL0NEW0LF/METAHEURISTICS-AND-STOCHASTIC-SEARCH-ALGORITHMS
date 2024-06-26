import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import utils.functions as functions

def func_plot(func_name, Best_pos, fitness_track, algo_name, Best_fit = None):
    fobj, lb, ub, dim = functions.GetFunctionsDetails(func_name)

    if func_name in ['F1', 'F2', 'F3', 'F4', 'F6', 'F14']:
        x = y = np.arange(-100, 101, 2)
    elif func_name == 'F5':
        x = y = np.arange(-200, 201, 2)
    elif func_name == 'F7':
        x = y = np.arange(-1, 1.01, 0.03)
    elif func_name in ['F8', 'F11']:
        x = y = np.arange(-500, 501, 10)
    elif func_name in ['F9', 'F15', 'F17', 'F19', 'F20', 'F21', 'F22', 'F23']:
        x = y = np.arange(-5, 5.1, 0.1)
    elif func_name == 'F10':
        x = y = np.arange(-20, 20.1, 0.5)
    elif func_name == 'F12':
        x = y = np.arange(-10, 10.1, 0.1)
    elif func_name == 'F13':
        x = y = np.arange(-5, 5.01, 0.08)
    elif func_name == 'F16':
        x = y = np.arange(-1, 1.01, 0.01)
    elif func_name == 'F18':
        x = y = np.arange(-5, 5.01, 0.06)
    elif func_name in ['F24', 'F25', 'F26', 'F27', 'F28', 'F29', 'F30', 'F31', 'F32', 'F33', 'F35', 'F36', 'F37', 'F38']:
        x = y = np.arange(-100, 101, 2)

    L = len(x)
    f = np.zeros((L, L))

    for i in range(L):
        for j in range(L):
            if func_name not in ['F15', 'F19', 'F20', 'F21', 'F22', 'F23']:
                f[i, j] = fobj([x[i], y[j]])
            if func_name == 'F15':
                f[i, j] = fobj([x[i], y[j], 0, 0])
            if func_name == 'F19':
                f[i, j] = fobj([x[i], y[j], 0])
            if func_name == 'F20':
                f[i, j] = fobj([x[i], y[j], 0, 0, 0, 0])
            if func_name in ['F21', 'F22', 'F23']:
                f[i, j] = fobj([x[i], y[j], 0, 0])

    fig = go.Figure(data=[go.Surface(z=f, x=x, y=y, colorscale="Reds", opacity=0.5)])
    if Best_fit is not None:
        fig.add_trace(go.Scatter3d(x=[Best_pos[0]], y=[Best_pos[1]], z=[Best_fit], mode='markers', marker=dict(color='blue', size=5), name=f'Optimum Found using {algo_name}'))
    else:
        fig.add_trace(go.Scatter3d(x=[Best_pos[0]], y=[Best_pos[1]], z=[fobj(Best_pos)],
                               mode='markers', marker=dict(color='blue', size=5), name=f'Optimum Found using {algo_name}'))
    fig.update_layout(
        title= f'Plot for function {func_name} and optimum found using {algo_name}',
        autosize=False,
        width=2000, 
        height=2000,
        margin=dict(l=65, r=50, b=65, t=90), 
        scene_aspectmode='cube',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        )
    )
    fig.show()

    fig = go.Figure(data=[go.Scatter(x=list(range(len(fitness_track))), y=fitness_track, mode='lines', name='Fitness')])

    fig.update_layout(
        title= f'Plot for function {func_name} and optimum found using {algo_name}',
        autosize=False,
        width=2000, 
        height=1000,
        margin=dict(l=65, r=50, b=65, t=90), 
        scene_aspectmode='cube',
    )

    fig.show()