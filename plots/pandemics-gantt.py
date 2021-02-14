import plotly.figure_factory as ff

data = [
    ("Odra", 12, 18),
    ("Ospa wietrzna (chickenpox)", 10, 12),
    ("Świnka", 10, 12),
    ("Różyczka", 6, 7),
    ("Polio", 5, 7),
    ("Ospa wietrzna (smallpox)", 3.5, 6),
    ("HIV", 2, 5),
    ("Zwykłe przeziębienie", 2, 3),
    ("Błonica", 1.7, 4.3),
    ("Ebola", 1.5, 1.9),
    ("Grypa (1918)", 1.4, 2.8),
    ("Grypa (2009)", 1.2, 1.6),
    ("SARS", 0.19, 1.1),
    ("Grypa sezonowa", 0.9, 2.1),
    ("COVID-19", 0.8, 5.7),
    ("MERS", 0.3, 0.8),
]

plotly_input = [
    {"Task": disease, "Start": min_r, "Finish": max_r}
    for disease, min_r, max_r in data
]

fig = ff.create_gantt(plotly_input)
fig.update_layout(
    width=1024,
    xaxis_type="linear",
    xaxis_showgrid=True,
    template="simple_white",
    font_family="Times New Roman",
    xaxis_title="Basic Reproduction Number",
)
fig.show()
