
PYTHON_ADD_MODULE(meshrender vstsim/meshrender.cpp)
#FILE(COPY render_mesh.py DESTINATION .)
INSTALL(TARGETS meshrender DESTINATION dexnet)
