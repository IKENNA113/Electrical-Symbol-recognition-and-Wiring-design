import cv2
import time
import numpy as np
import os
import json




SCALE = 10

#class imports
class Component:

    def __init__(self, name, location,routeId = 0):
        self.name = name
        self.location = location
        self.centerPoint = (int((self.location[0]+self.location[2])/2), int((self.location[1]+self.location[3])/2))
        self.edgePointTop = None
        self.edgePointBottom = None
        self.edgePointLeft = None
        self.edgePointRight = None
        self.routeId = routeId
        self.calEdgePoints()
        color = np.random.randint(0, 255, 3)
        color = tuple(color)
        self.color = color
        self.pannelConnection = False

    def __str__(self):
        return f"{self.name} - {self.description} - {self.price}"

    def calEdgePoints(self):
        #calculate the center edge point
        self.edgePointTop = (int((self.location[0]+self.location[2])/2), self.location[1])
        self.edgePointBottom = (int((self.location[0]+self.location[2])/2), self.location[3])
        self.edgePointLeft = (self.location[0], int((self.location[1]+self.location[3])/2))
        self.edgePointRight = (self.location[2], int((self.location[1]+self.location[3])/2))
        return True

class Router:
    def __init__(self):
        self.routes = {}
        self.plainX = 11520
        self.plainY = 8320
        self.grid = self.createGrid()
        self.obstacleImage = None

    #create a grid image with plain dimension with 30x30 grid
    def createGrid(self):
        grid = np.zeros((self.plainY, self.plainX, 3), dtype=np.uint8)
        #create a grid
        for i in range(0, self.plainY, 30):
            for j in range(0, self.plainX, 30):
                #set random color
                # grid[i:i+30, j:j+30] = np.random.randint(0, 255, 3)
                #set white color
                grid[i:i+30, j:j+30] = [0, 255, 0]
        return grid





    def routeTheGridAstar(self, componentList,image,Obmap = None):
        map = Map(self.plainX,self.plainY)
        map.create_nodes(map.image)
        map.setObstacleMap(Obmap)
        map.create_route_map(self.plainY, self.plainX)
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        #route the grid
        #separate the pannels and symbols
        nearest_points = []
        routeComponentList = {}
        dataWriter = ""

        for comp in componentList:
            if comp.routeId not in routeComponentList:
                routeComponentList[comp.routeId] = []
            routeComponentList[comp.routeId].append(comp)

        completeDistance = 0
        for routingIds in routeComponentList.keys():



            for comp2 in componentList:
                if comp2.routeId == routingIds and comp2.name == "PANELBOARD":
                    pannelLocation = comp2.location
                    pannelCenterPoint = comp2.centerPoint
                    color = comp2.color
                    break


            #sort the component list based on the distance from the center point
            routeComponentList[routingIds] = sorted(routeComponentList[routingIds], key=lambda x: self.distanceCalculate(x.centerPoint, pannelCenterPoint))
            totalBranchDistance = 0
            for comp in routeComponentList[routingIds]:

                routeId = comp.routeId
                location = comp.location
                centerPoint = comp.centerPoint
                pannelLocation = None






                #update the destination with the nearest point
                destination_point = pannelCenterPoint
                # print("Center Point: ", centerPoint, "Pannel Center Point: ", pannelCenterPoint)

                #connect the closest pannel with the symbol
                lowestDistance = 999999999
                for comp2 in routeComponentList[routeId]:
                    #find the nearest point to center point
                    if comp2.centerPoint != centerPoint:
                        distance = self.distanceCalculate(centerPoint, comp2.centerPoint)
                        print("Distance: ", distance, comp2.centerPoint, centerPoint, comp2.pannelConnection, lowestDistance)
                        if distance < lowestDistance and comp2.pannelConnection == True:
                            lowestDistance = distance
                            destination_point = comp2.centerPoint

                # if lowestDistance == 999999999:
                #     print("No pannel connected")
                comp.pannelConnection = True


                #print update routeComponentList
                for tempRouteId in routeComponentList[routeId]:
                    print(tempRouteId.routeId, tempRouteId.pannelConnection)


                #draw the lines with multiline plotter
                points = [centerPoint, destination_point]
                # print(centerPoint, pannelCenterPoint)
                #convert the points as numpy array
                print(points)
                path = map.astar(map.pathFindingArray, (centerPoint[0]//10, centerPoint[1]//10), (destination_point[0]//10, destination_point[1]//10))
                print("Path: ", path)

                #diagonal line detection and the starting and ending point
                newPath = []
                if path != None:
                    startingPointOfDiagonal = None
                    endingPointOfDiagonal = None
                    for i in range(len(path)-1):
                        if path[i][0] != path[i+1][0] and path[i][1] != path[i+1][1] and startingPointOfDiagonal == None:
                            startingPointOfDiagonal = path[i]
                            newPath.append(startingPointOfDiagonal)

                        elif (path[i][0] == path[i+1][0] or path[i][1] == path[i+1][1]) and startingPointOfDiagonal != None:
                            endingPointOfDiagonal = path[i]

                            print("Starting Point: ", startingPointOfDiagonal, "Ending Point: ", endingPointOfDiagonal)
                            #convert the diagonal into straight line with mid point
                            x1, y1 = startingPointOfDiagonal
                            x2, y2 = endingPointOfDiagonal
                            mid_x = (x1 + x2) // 2
                            mid_y = (y1 + y2) // 2
                            newPath.insert(i+1, [mid_x, y1])
                            newPath.insert(i+2, [mid_x, mid_y])
                            newPath.insert(i+3, [x2, mid_y])
                            newPath.insert(i+4, endingPointOfDiagonal)
                            startingPointOfDiagonal = None
                            endingPointOfDiagonal = None


                        if startingPointOfDiagonal == None:
                            newPath.append(path[i])

                path = newPath
                print("Path: ", path)







                #distance calculation
                if path != None:
                    branch_distance = len(path)*10
                    completeDistance += branch_distance
                    totalBranchDistance += branch_distance
                else:
                    branch_distance = 0
                    completeDistance += branch_distance
                    totalBranchDistance += branch_distance
                dataWriter += f"RouteId: {routeId} - Branch Distance: {branch_distance} - Symbol point: {centerPoint} - Power Point: {destination_point} - Total Distance till now: {totalBranchDistance}\n"
                #trace the path with nodes
                pathNodeList = []
                for i in path:
                    node = map.nodeLocator(i[0], i[1])
                    # print("position", node.position)
                    pathNodeList.append(node)
                # print("pathNodeList", pathNodeList)
                map.image = image.copy()
                image = map.connectInputNodes(pathNodeList, (int(color[0]), int(color[1]), int(color[2])))

                #show the image in cv2
                # cv2.imshow("image", image)
                #save the image
                cv2.imwrite("output/astar.jpg", image)
                cv2.waitKey(1)


        #add the complete distance to the data writer
        dataWriter += f"complete distance: {completeDistance}\n"

        # save the data
        with open("output/data.txt", "w") as file:
            file.write(dataWriter)
        return self.grid, image,dataWriter


    def distanceCalculate(self,p1, p2):
        """p1 and p2 in format (x1,y1) and (x2,y2) tuples"""
        dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        return dis

    def setObstracleImage(self,image):
        self.obstacleImage = image
        return True


    #calculate the closest pannel location
    def calculateClosestPannel(self, componentList):
        #calculate the closest pannel near the symbol

        #separate the pannels and symbols
        pannels = []
        symbols = []
        for comp in componentList:
            print(comp.name, comp.location, comp.routeId)
            if comp.name == "PANELBOARD":
                pannels.append(comp)
            else:
                symbols.append(comp)


        #calculate the distance between the symbol and the nearest pannel

        for symbol in symbols:
            minDistance = 999999999
            minPannel = None
            for pannel in pannels:
                distance = self.distanceCalculate(symbol.centerPoint, pannel.centerPoint)
                print(distance, minDistance, pannel.routeId, symbol.location, pannel.location)
                if distance < minDistance:
                    minDistance = distance
                    minPannel = pannel

            symbol.routeId = minPannel.routeId

        #merge the pannels and symbols
        # print(pannels + symbols)
        return pannels + symbols


    def getDirection(self, sourceLocation, destinationLocation):
        #get the direction

        #move through all four direction and calculate the distance
        #get the distance for all four direction with 30x30 grid
        distance = {}
        updatedSourceLocation = None
        distance["UP"] = self.distanceCalculate((sourceLocation[0], sourceLocation[1]-30), destinationLocation)
        distance["DOWN"] = self.distanceCalculate((sourceLocation[0], sourceLocation[1]+30), destinationLocation)
        distance["LEFT"] = self.distanceCalculate((sourceLocation[0]-30, sourceLocation[1]), destinationLocation)
        distance["RIGHT"] = self.distanceCalculate((sourceLocation[0]+30, sourceLocation[1]), destinationLocation)

        #get the minimum distance
        minDistance = min(distance.values())

        #get the direction
        direction = [key for key in distance if distance[key] == minDistance]
        #get the updated source location
        if direction[0] == "UP":
            updatedSourceLocation = (sourceLocation[0], sourceLocation[1]-30)
        elif direction[0] == "DOWN":
            updatedSourceLocation = (sourceLocation[0], sourceLocation[1]+30)
        elif direction[0] == "LEFT":
            updatedSourceLocation = (sourceLocation[0]-30, sourceLocation[1])
        elif direction[0] == "RIGHT":
            updatedSourceLocation = (sourceLocation[0]+30, sourceLocation[1])
        else:
            print("Invalid direction")

        return direction[0], updatedSourceLocation

    def drawLS(self,image, sourceLocation, direction, length = 30):
        #derive the 30x30 grid with source location x1,y1 and destination location x1+30, y1+30

        gridImg = np.ones((30, 30, 3), dtype=np.uint8)*255

        #draw a line with 10 pixel width from middle 10 pixel for top, bottom, left and right
        if direction == "UP":
            gridImg = cv2.line(gridImg, (15, 0), (15, 30), (0, 0, 0), 10)
        elif direction == "DOWN":
            gridImg = cv2.line(gridImg, (15, 0), (15, 30), (0, 0, 0), 10)
        elif direction == "LEFT":
            gridImg = cv2.line(gridImg, (0, 15), (30, 15), (0, 0, 0), 10)
        elif direction == "RIGHT":
            gridImg = cv2.line(gridImg, (0, 15), (30, 15), (0, 0, 0), 10)
        else:
            print("Invalid direction")

        #apply the grid to the image
        image[sourceLocation[0]:sourceLocation[0]+30, sourceLocation[1]:sourceLocation[1]+30] = gridImg

        return image

    def multilinePlotter(self,image, points, nearest_points,color=(80,120,90)):
        for i in range(len(points) - 1):
            # print(i, points[i])
            x1, y1 = points[i]
            x2, y2 = points[i + 1]

            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2

            #add the nearest points
            nearest_points.append([x1, y1])
            nearest_points.append([mid_x, y1])
            nearest_points.append([mid_x, mid_y])
            nearest_points.append([x2, mid_y])

            #line distance calculator for the bend line
            branch_distance = self.distanceCalculate([x1, y1], [mid_x, y1]) + self.distanceCalculate([mid_x, y1], [mid_x, mid_y]) + self.distanceCalculate([mid_x, mid_y], [x2, mid_y]) + self.distanceCalculate([x2, mid_y], [x2, y2])
            # branch_distance = (self.distanceCalculate([x1, y1], [mid_x, y1]) , self.distanceCalculate([mid_x, y1], [mid_x, mid_y]) , self.distanceCalculate([mid_x, mid_y], [x2, mid_y]) , self.distanceCalculate([x2, mid_y], [x2, y2]))

            print("color", color)
            # Draw horizontal and vertical lines to create a 90-degree bend
            cv2.line(image, (x1, y1), (mid_x, y1), color, 2)

            cv2.line(image, (mid_x, y1), (mid_x, mid_y), color, 2)
            cv2.line(image, (mid_x, mid_y), (x2, mid_y), color, 2)
            cv2.line(image, (x2, mid_y), (x2, y2), color, 2)

        return image,branch_distance

class Node:
    def __init__(self, x, y,parent=None, position=None):
        self.x = x
        self.y = y
        self.neighbours = []
        self.visited = False
        self.parent = None
        self.distance = 0
        self.cost = 0
        self.heuristic = 0

        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def add_neighbour(self, node):
        self.neighbours.append(node)

    def __str__(self):
        return f'Node {self.position}'

    def __eq__(self, other):
        # print("Comparing nodes", self.position, other.position)
        return self.position == other.position

class Map:
    def __init__(self,width,height) -> None:
        self.height = height
        self.width = width
        self.channels = 3
        self.image = None
        self.create_map(height, width, 3)
        self.nodes = []
        self.pathFindingArray = []
        self.obstacleMap = None
        # self.create_route_map(height, width)

    def setObstacleMap(self,obMap):
        # obMap = cv2.cvtColor(obMap, cv2.COLOR_BGR2GRAY)
        self.obstacleMap = obMap
        return True

    def create_route_map(self, height, width):
        pathFindingArray = []
        for i in range(width//SCALE):
            pathFindingArray.append([])
            for j in range(height//SCALE):
                # check for white and assign zeros
                # print("Checking for obstacle", i*SCALE, j*SCALE, self.obstacleMap[i*SCALE:i*SCALE+SCALE,j*SCALE:j*SCALE+SCALE].sum())
                # print("shape of obstacle map", self.obstacleMap[i*SCALE:i*SCALE+SCALE,j*SCALE:j*SCALE+SCALE].shape)
                if self.obstacleMap[i*SCALE:i*SCALE+SCALE,j*SCALE:j*SCALE+SCALE].sum() >= 0:
                    pathFindingArray[i].append(0)
                else:
                    # print("Checking for obstacle", i*SCALE, j*SCALE, self.obstacleMap[i*SCALE:i*SCALE+SCALE,j*SCALE:j*SCALE+SCALE].sum())
                    # print("shape of obstacle map", self.obstacleMap[i*SCALE:i*SCALE+SCALE,j*SCALE:j*SCALE+SCALE].shape)
                    # print("Adding 1", i, j)
                    # exit()
                    pathFindingArray[i].append(1)
                # pathFindingArray[i].append(0)
        self.pathFindingArray = pathFindingArray
        return True

    def create_map(self, height, width, channels):
        image = np.zeros((height, width, channels), dtype=np.uint8)
        self.image = image
        self.height = height
        self.width = width
        self.channels = channels
        return image

    def create_nodes(self, image):
        print("Creating nodes")
        print("image shape", image.shape[0] // SCALE, image.shape[1] // SCALE)
        nodes = []
        for i in range((image.shape[1] // SCALE)+1):
            for j in range((image.shape[0] // SCALE)+1):
                node = Node(x = i * SCALE, y = j * SCALE, parent=None, position=(i * SCALE, j * SCALE))
                nodes.append(node)
        self.nodes = nodes
        return nodes

    def connect_nodes(self):
        for node in self.nodes:
            for neighbour in self.nodes:
                if node != neighbour:
                    distance = ((neighbour.x - node.x) ** 2 + (neighbour.y - node.y) ** 2) ** 0.5
                    if distance < 31:
                        node.add_neighbour(neighbour)
        return nodes

    def draw_map(self):
        for node in self.nodes:
            for neighbour in node.neighbours:
                cv2.line(self.image, (node.x, node.y), (neighbour.x, neighbour.y), (255, 255, 255), 2)
            cv2.circle(self.image, (node.x, node.y), 5, (0, 0, 255), -1)

        return self.image

    def connectInputNodes(self,nodes,color=(0, 255, 0)):
        #draw a line connecting all the nodes in the list
        for i in range(len(nodes)-1):
            # print("Connecting nodes", nodes[i].position, nodes[i+1].position)
            cv2.line(self.image, (nodes[i].x, nodes[i].y), (nodes[i+1].x, nodes[i+1].y), color, 2)
        return self.image

    def nodeLocator(self, x, y):
        x = x * SCALE
        y = y * SCALE
        for node in self.nodes:
            if x - node.x < SCALE and y - node.y < SCALE:
                return node
        return None

    def astar(self,maze, start, end):
        """Returns a list of tuples as a path from the given start to the given end in the given maze"""

        # Create start and end node
        start_node = Node(x=0,y=0,parent=None, position=start)
        start_node.g = start_node.h = start_node.f = 0
        end_node = Node(x=0,y=0,parent=None, position=end)
        end_node.g = end_node.h = end_node.f = 0

        # Initialize both open and closed list
        open_list = []
        closed_list = []

        # Add the start node
        open_list.append(start_node)

        # Loop until you find the end
        while len(open_list) > 0:

            # Get the current node
            current_node = open_list[0]
            current_index = 0
            for index, item in enumerate(open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index

            # Pop current off open list, add to closed list
            open_list.pop(current_index)
            closed_list.append(current_node)

            # Found the goal
            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                return path[::-1] # Return reversed path

            # Generate children
            children = []
            for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares
            #add only top, bottom, left and right
            # for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:

                # Get node position
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                # Make sure within range
                if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                    continue

                # Make sure walkable terrain
                if maze[node_position[0]][node_position[1]] != 0:
                    continue

                # Create new node
                new_node = Node(x=0,y=0,parent=current_node,position=node_position)

                # Append
                children.append(new_node)

            # Loop through children
            for child in children:

                # Child is on the closed list
                for closed_child in closed_list:
                    if child == closed_child:
                        continue

                # Create the f, g, and h values
                child.g = current_node.g + 1
                child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
                child.f = child.g + child.h

                # Child is already in the open list
                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        continue

                # Add the child to the open list
                open_list.append(child)

class Obstaclemap:
    def __init__(self, image):
        self.image = image
        self.thresh = self.contourDet(image)
        self.border = self.borderSegmentation(self.thresh)

    def contourDet(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # add black border around threshold image to avoid corner being largest distance
        gray = cv2.copyMakeBorder(gray, 1,1,1,1, cv2.BORDER_CONSTANT, (0))
        w,h = gray.shape

        lower = 160 # threshold: tune this number to your needs
        upper = 220



        #bgr threshold
        lower = np.array([lower,lower,lower])
        upper = np.array([upper,upper,upper])

        # Threshold
        thresh = cv2.inRange(image, lower, upper)

        thresh = cv2.medianBlur(thresh, 3)

        return thresh

    def borderSegmentation(self,image):
        #apply median filter
        image = cv2.medianBlur(image, 3)
        image = cv2.medianBlur(image, 3)

        #Convert the image from gray to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        #based on the border, segment the image



        #convert the image to gray
        outputImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        #find the contours
        #invert the image
        # outputImage = 255 - outputImage
        contours, _ = cv2.findContours(outputImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #find the largest contour
        maxArea = 0
        print(len(contours),contours[0])
        for id,con in enumerate(contours):
            area = cv2.contourArea(con)
            if area > maxArea:
                maxArea = area
                largest = id

        print("Largest contour", largest," Its area", maxArea)
        # print(c.shape)
        # exit()


        #convert the image to BGR
        outputImage = cv2.cvtColor(outputImage, cv2.COLOR_GRAY2BGR)

        # #draw the largest contour
        outputImage = cv2.drawContours(outputImage, contours, largest, (255, 0, 0), -1)

        # dilate the image
        kernel = np.ones((10, 10), np.uint8)

        outputImage = cv2.dilate(outputImage, kernel, iterations=15)


        #fill all the contours except the largest except the largest contour with black
        for i in range(len(contours)):
            if i != largest:
                outputImage = cv2.drawContours(outputImage, contours, i, (0, 255, 0), -1)



        #second step
        #convert the image to gray
        outputImage = cv2.cvtColor(outputImage, cv2.COLOR_RGB2GRAY)
        #find the contours
        contours, _ = cv2.findContours(outputImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #find the largest contour
        maxArea = 0
        print(len(contours),contours[0])
        for id,con in enumerate(contours):
            area = cv2.contourArea(con)
            if area > maxArea:
                maxArea = area
                largest = id

        print("Largest contour", largest," Its area", maxArea)
        # print(c.shape)
        # exit()


        #convert the image to BGR
        outputImage = cv2.cvtColor(outputImage, cv2.COLOR_GRAY2BGR)

        # #draw the largest contour
        outputImage = cv2.drawContours(outputImage, contours, largest, (255, 0, 0), -1)


        #resize it to a smaller size
        outputImage = cv2.resize(outputImage, (int(outputImage.shape[1]/10), int(outputImage.shape[0]/10)))


        #fill black for all pixel except blue
        for i in range(outputImage.shape[0]):
            for j in range(outputImage.shape[1]):
                if outputImage[i, j][0] == 255 and outputImage[i, j][1] == 0 and outputImage[i, j][2] == 0:
                    # outputImage[i, j] = [0, 0, 0]
                    pass
                else:
                    outputImage[i, j] = [0, 0, 0]

        #resize it to the original size
        outputImage = cv2.resize(outputImage, (image.shape[1], image.shape[0]))

        #dilate the image
        kernel = np.ones((10, 10), np.uint8)
        outputImage = cv2.dilate(outputImage, kernel, iterations=30)




        #convert the image into gray
        outputImage = cv2.cvtColor(outputImage, cv2.COLOR_BGR2GRAY)

        #convert the high val pix to white
        outputImage[outputImage > 0] = 255

        return outputImage





if __name__ == "__main__":

    #create a router instance
    router = Router()

    WIDTH =  11520
    HEIGHT = 8320

    PATH = os.getcwd()+'/input1/Diagrams_images_anotation_1/actual_Reconstructed_panel/'


    #output folder creation
    os.makedirs(os.getcwd()+'/output',exist_ok=True)


    for file in os.listdir(PATH):
        if file.endswith(".png"):
            start_time = time.time()
            # image = cv2.imread('/home/countai/Documents/is/image_construction/input/Diagrams_images_anotation_1/original image_8320X11520/BR.E2.1.png')
            image = cv2.imread(PATH+'/'+file)
            image  = cv2.resize(image,(WIDTH,HEIGHT))

            #creating obstacle map
            obstacleMap = Obstaclemap(image.copy())

            #save the obstacle map in output folder
            cv2.imwrite(os.getcwd()+'/output/'+file[:-4]+'_obstacle.png',obstacleMap.border)


            #open the file with class names
            with open('/home/ikenna/Downloads/code_to_publish/input_copy/Diagrams_images_anotation_1/Class_mapping.txt') as f:
                lines = f.read()
                id_to_name = {}
                #conver the text as json
                id_to_name = json.loads(lines)
                #get the class names)

            #components list
            components = []
            routeId = 0
            #read the yolo coordinates
            with open(PATH+'/'+file[:-4]+'.txt') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split()
                    # print(line)
                    classId = int(line[0])
                    x = float(line[1])
                    y = float(line[2])
                    w = float(line[3])
                    h = float(line[4])
                    # print(classId)
                    # print(x,y,w,h)
                    l = int((x - w / 2) * WIDTH)
                    r = int((x + w / 2) * WIDTH)
                    t = int((y - h / 2) * HEIGHT)
                    b = int((y + h / 2) * HEIGHT)

                    if l < 0:
                        l = 0
                    if r > WIDTH - 1:
                        r = WIDTH - 1
                    if t < 0:
                        t = 0
                    if b > HEIGHT - 1:
                        b = HEIGHT - 1
                    #plot the rectangle with condition
                    if classId == 14:

                        #plot the rectangle
                        # cv2.rectangle(image, (l, t), (r, b), (0, 255, 0), 2)
                        #draw a circle
                        cv2.circle(image, (int((l + r) / 2), int((t + b) / 2)), 5, (0, 255, 0), -1)
                        #put the text
                        # cv2.putText(image, id_to_name[str(classId)], (l, t - 7), cv2.FONT_HERSHEY_COMPLEX, 0.37, (0, 0, 255), 1)
                        #create a component
                        conponentTemp = Component(id_to_name[str(classId)], (l, t, r, b),routeId=routeId)
                        routeId += 1
                        #plot the edge points
                        cv2.circle(image, conponentTemp.edgePointTop, 5, (201, 138, 99), -1)
                        cv2.circle(image, conponentTemp.edgePointBottom, 5, (201, 138, 99), -1)
                        cv2.circle(image, conponentTemp.edgePointLeft, 5, (201, 138, 99), -1)
                        cv2.circle(image, conponentTemp.edgePointRight, 5, (201, 138, 99), -1)

                        components.append(conponentTemp)

                    else:
                        #plot the rectangle
                        # cv2.rectangle(image, (l, t), (r, b), (0, 0, 255), 2)
                        #draw a circle
                        cv2.circle(image, (int((l + r) / 2), int((t + b) / 2)), 5, (0, 0, 255), -1)
                        #put the text
                        # cv2.putText(image, id_to_name[str(classId)], (l, t - 7), cv2.FONT_HERSHEY_COMPLEX, 0.37, (0, 0, 255), 1)

                        #create a component
                        conponentTemp = Component(id_to_name[str(classId)], (l, t, r, b))

                        #plot the edge points
                        cv2.circle(image, conponentTemp.edgePointTop, 5, (201, 138, 99), -1)
                        cv2.circle(image, conponentTemp.edgePointBottom, 5, (201, 138, 99), -1)
                        cv2.circle(image, conponentTemp.edgePointLeft, 5, (201, 138, 99), -1)
                        cv2.circle(image, conponentTemp.edgePointRight, 5, (201, 138, 99), -1)

                        components.append(conponentTemp)





            distanceOptimisedComponentList = router.calculateClosestPannel(components)

            for component in distanceOptimisedComponentList:
                cv2.putText(image, str(component.routeId), component.centerPoint, cv2.FONT_HERSHEY_COMPLEX, 0.60, (0, 0, 255), 1)



            #grid routing
            grid,image,datawrite = router.routeTheGridAstar(distanceOptimisedComponentList,image,Obmap=obstacleMap.border)

            end_time =time.time()
            processing_time = end_time - start_time
            print(f"Processing time for {file}: {processing_time} seconds")

            datawrite += f"Processing time for {file}: {processing_time} seconds\n"

            # create a named window
            # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            #display the image
            # cv2.imshow('image',image)
            # cv2.waitKey(0)

            #save the image in the output folder
            cv2.imwrite(os.getcwd()+'/output/'+file,image)

            #save the datawriter in the output folder
            with open(os.getcwd()+'/output/'+file[:-4]+'.txt','w') as f:
                f.write(datawrite)
