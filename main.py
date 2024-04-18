from itertools import pairwise

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from pydantic import BaseModel

from afsa import AFSA


class AfsaRequest(BaseModel):
    max_iter: int = 300
    N: int = 40
    v: float = 15
    theta: float = 0.95
    eta: float = 0.3
    k: float = 0.3


class FishUI(BaseModel):
    data: list[int]
    fitness: float


def main():
    st.title('Косяк рыб')

    max_iter = st.number_input('Количество итераций, max_iter', value=300, min_value=1)
    N = st.number_input('Количество рыб, N', value=40, min_value=1)
    v = st.number_input('Радиус визуального диапазона, v', value=15., min_value=0.)
    theta = st.slider('Параметр отношения заполненности, theta', value=0.95, min_value=0., max_value=1., step=0.01)
    eta = st.slider('eta', value=0.3, min_value=0., max_value=1.)
    k = st.slider('k', value=0.3, min_value=0., max_value=1.)
    file = st.file_uploader('Загрузить матрицу смежности', type='txt')

    if st.button('Решить'):
        if file is not None:
            data = [tuple(map(float, line.decode().strip('\r\n').split(', '))) for line in file.readlines()]
            afsa_params = AfsaRequest(max_iter=max_iter, N=N, v=v, theta=theta, eta=eta, k=k)

            result = AFSA.resolve(
                adjacency_matrix=data,
                **afsa_params.model_dump(),
            )

            for result_ in result:
                st.code(result_)
            
            gr = nx.Graph()
            for i, line in enumerate(data):
                for j, val in enumerate(line):
                    if val == float('inf'):
                        continue
                    gr.add_edge(i, j, weight=val)

            pos = nx.spring_layout(gr)
            weights = nx.get_edge_attributes(gr, 'weight')

            nx.draw_networkx_nodes(gr, pos)
            nx.draw_networkx_edges(gr, pos)
            nx.draw_networkx_labels(gr, pos)
            nx.draw_networkx_edge_labels(gr, pos, edge_labels=weights)

            answer = result[0].data
            answer = [*pairwise(answer), (answer[-1], answer[0])]

            nx.draw_networkx_edges(gr, pos, edgelist=answer, edge_color='red')
            
            plt.axis('off')
            st.pyplot(plt)


if __name__ == '__main__':
    main()
