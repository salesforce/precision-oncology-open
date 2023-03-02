def imageData = QPEx.getCurrentImageData()
image_name = imageData.toString().split(" ").last().split("\\.").first() + ".csv"
def path = buildFilePath("/export/home/nuclei_coords/raw_annos/", image_name)
def file = new File(path)

def i = 0

def curr_coord_string = "" 
for (pathObject in getDetectionObjects()) {
    // Check for interrupt (Run -> Kill running script)
    if (Thread.interrupted())
        break
    // Get the ROI
    def roi = pathObject.getROI()
    if (roi == null)
        continue
//    print roi
    x_coords = []
    y_coords = []
    for (point in roi.getAllPoints()){
        x_coords.add(point.x)
        y_coords.add(point.y)
    }

    def min_x = x_coords.min().toInteger()
    def min_y = y_coords.min().toInteger()
    def max_x = x_coords.max().toInteger()
    def max_y = y_coords.max().toInteger()

    // Write the points; but beware areas, and also ellipses!
    def str = "${-> min_x}, ${->min_y}, ${->max_x}, ${->max_y}\n"
    curr_coord_string = curr_coord_string + str
    i = i + 1
    if (i % 500 == 0){
        file << curr_coord_string
        curr_coord_string = ""
    }
}
file << curr_coord_string
