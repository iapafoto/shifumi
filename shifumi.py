import glfw
from OpenGL.GL import *
import numpy as np
import cv2
import mediapipe as mp
import time
import math

# === Initialisation MediaPipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.2, min_tracking_confidence=0.2)


finger_tips = {
    'thumb': [1, 2, 3, 4],
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'little': [17, 18, 19, 20]
}

# === GLSL Shader ===

def read_shader_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# === OpenGL Helpers ===
def compile_shader(shader_code, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, shader_code)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader

def create_shader_program():
    vertex_shader_code = read_shader_file("vertex_shader.glsl")
    fragment_shader_code = read_shader_file("fragment_shader.glsl")
    
    vertex_shader = compile_shader(vertex_shader_code, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_shader_code, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(program))
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return program
    
    
def calculate_distance_3d(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)


def mix(x, y, a):
    return (1 - a) * x + a * y
    
def inv_mix(v0, v1, v):
    if v1 == v0: 
        return 1
    else:
        return (v - v0) / (v1 - v0)


# Fonction pour calculer l'indice d'ouverture des doigts
def calculate_finger_opening(i, hand_landmarks, finger_tips):
    tip_index = finger_tips[-1]  # Bout du doigt
    tip = hand_landmarks.landmark[tip_index]  # Coordonnées du bout du doigt
    pt0 = hand_landmarks.landmark[0]  # Base de la main (point 0)
    pt9 = hand_landmarks.landmark[9]  # Point de référence (point 9)

    # Coordonnées 3D des points (x, y, z) en "world"
    tip_coords = (tip.x, tip.y, tip.z)
    pt0_coords = (pt0.x, pt0.y, pt0.z)
    pt9_coords = (pt9.x, pt9.y, pt9.z)

    # Calculer la distance entre le bout du doigt et la base de la main
    distance = calculate_distance_3d(pt0_coords, tip_coords)

    # Calculer la distance entre la base de la main et le point 9
    ref_distance = calculate_distance_3d(pt0_coords, pt9_coords)

    # Calculer l'indice d'ouverture selon la formule donnée
    #opening = (distance / ref_distance) * 0.8 - 0.5
    opening = (distance / ref_distance);
    
    if i==1:
        opening = inv_mix(0.9,1.9,opening)
    elif i==2:
        opening = inv_mix(0.8,1.97,opening)
    elif i==3:
        opening = inv_mix(0.65,1.85,opening)
    elif i==4:
        opening = inv_mix(0.5,1.5,opening)
    elif i==0:
        opening = opening-.5
    opening = max(0.0, min(opening, 1.0))
    return opening, pt0_coords, tip_coords

def initialize_opengl(window):
    # Configuration du contexte OpenGL
    glfw.make_context_current(window)

    # Création du programme de shaders
    shader_program = create_shader_program()  # Vous pouvez utiliser votre propre fonction pour créer le shader
    glUseProgram(shader_program)

    # Création d'un VBO pour un rectangle plein écran
    vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0,
    ], dtype=np.float32)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # Création d'un VAO
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * vertices.itemsize, None)

    return shader_program, vao


def send_data_to_shader(shader_program, finger_positions, finger_openings, finger_speeds, pt0_speed, time):
    global t0
    for i in range(5):
        glUniform3f(glGetUniformLocation(shader_program, f"finger_positions[{i}]"), *finger_positions[i])
        glUniform1f(glGetUniformLocation(shader_program, f"finger_openings[{i}]"), finger_openings[i])
        glUniform1f(glGetUniformLocation(shader_program, f"finger_speeds[{i}]"), finger_speeds[i])
    glUniform1f(glGetUniformLocation(shader_program, "pt0_speed"), pt0_speed)
    glUniform1f(glGetUniformLocation(shader_program, "time"), time)


def display_finger_data(frame, finger_openings, finger_speeds, pt0_speed):
    y_offset = 100
    for i, finger in enumerate(finger_tips.keys()):
        opening = finger_openings[i]
        speed = 1000 * finger_speeds[i]
        if 1000 * pt0_speed > 30:
            color = (128, 128, 128)
        elif 1000 * pt0_speed < -30:
            color = (128, 0, 128)
        else:
            if speed > 40:
                color = (0, 0, 255)
            elif speed < -40:
                color = (255, 0, 0)
            else:
                color = (0, 0, 0)
        cv2.putText(frame, f"{finger}: {opening:.2f} (Speed: {speed:.2f})",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += 30

# === OpenCV + OpenGL Integration ===
def main():

    t0 = time.time()  # Retourne le temps actuel en secondes sous forme de float

    # Initialisation OpenGL
    if not glfw.init():
        raise Exception("GLFW can't be initialized!")
    
    window = glfw.create_window(800, 600, "OpenGL Shader + Webcam", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window can't be created!")

    # Initialisation OpenGL (contexte, shader, buffers)
    shader_program, vao = initialize_opengl(window)

    # Webcam capture avec OpenCV
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Webcam not accessible!")

    robot_finger_openings = [0.0] * 5
    finger_openings = [0.0] * 5
    finger_speeds = [0.0] * 5
    finger_positions = [(0.0, 0.0, 0.0)] * 5
    pt0_speed = 0.0
    pt0_position = (0.0, 0.0, 0.0)
    inertia_coefficient = 0.8

    while not glfw.window_should_close(window):
        ret, frame = cap.read()
        if not ret:
            break

        # Reconnaissance des mains
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calcul des positions, vitesses et ouvertures
                for i, finger in enumerate(finger_tips.keys()):
                    tip_index = finger_tips[finger][-1]
                    tip = hand_landmarks.landmark[tip_index]
                    tip_coords = (tip.x, tip.y, tip.z)

                    opening, base_pos, _ = calculate_finger_opening(i, hand_landmarks, finger_tips[finger])
                    finger_positions[i] = tip_coords
                    previous_opening = finger_openings[i]
                    speed = mix(opening - previous_opening, finger_speeds[i], inertia_coefficient)
                    finger_openings[i] = opening
                    finger_speeds[i] = speed

                    # Afficher l'indice d'ouverture et la vitesse du doigt sur la webcam
                    cv2.putText(frame, f"{finger}", (int(tip_coords[0] * frame.shape[1]), int(tip_coords[1] * frame.shape[0]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.line(frame,
                             (int(base_pos[0] * frame.shape[1]), int(base_pos[1] * frame.shape[0])),
                             (int(tip_coords[0] * frame.shape[1]), int(tip_coords[1] * frame.shape[0])),
                             (255 - 255 * opening, 255 * opening, 0), 2)

                # Vitesse du point 0
                pt0 = hand_landmarks.landmark[0]
                pt0_coords = (pt0.x, pt0.y, pt0.z)
                dx, dy, dz = pt0_coords[0] - pt0_position[0], pt0_coords[1] - pt0_position[1], pt0_coords[2] - pt0_position[2]
                instantaneous_speed = math.sqrt(dx**2 + dy**2 + dz**2)
                pt0_speed = mix(instantaneous_speed, pt0_speed, inertia_coefficient)
                pt0_position = pt0_coords
                

        display_finger_data(frame, finger_openings, finger_speeds, pt0_speed)

        # Envoyer les données au shader via des uniformes
        robot_inertia = 0.3
        for i in range(5):
            op = 1.
            if i==0:
                op = finger_openings[i]
            elif i==1:
                op = 1-abs(mix(finger_openings[1],finger_openings[2],.3)-mix(finger_openings[3],finger_openings[4],.3))
            elif i==2:
                op = 1-abs(mix(finger_openings[1],finger_openings[2],.7)-mix(finger_openings[3],finger_openings[4],.7))
            elif i==3:
                op = 1-mix(finger_openings[1],finger_openings[2],.3)
            elif i==4:
                op = 1-mix(finger_openings[1],finger_openings[2],.7)
            robot_finger_openings[i] = mix(op,robot_finger_openings[i], robot_inertia)
        
        send_data_to_shader(shader_program, finger_positions, robot_finger_openings, finger_speeds, pt0_speed, (time.time()-t0))

        # Rendu OpenGL
        glClear(GL_COLOR_BUFFER_BIT)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glfw.swap_buffers(window)

        # Affichage OpenCV
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        glfw.poll_events()

    cap.release()
    cv2.destroyAllWindows()
    glfw.terminate()

if __name__ == "__main__":
    main()
