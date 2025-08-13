import urllib.request
import gzip
import os

def download_mnist_via_keras():
    try:
        import tensorflow as tf
        print("Using TensorFlow/Keras to download MNIST...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        print("Converting Keras format to CSV...")
        print("Converting training data...")
        with open("mnist_train.csv", "w") as f:
            for i in range(len(x_train)):
                label = y_train[i]
                pixels = x_train[i].flatten()
                line = f"{label}," + ",".join(str(p) for p in pixels) + "\n"
                f.write(line)
                if (i + 1) % 10000 == 0:
                    print(f"  Processed {i + 1}/{len(x_train)} training samples")
        print("Converting test data...")
        with open("mnist_test.csv", "w") as f:
            for i in range(len(x_test)):
                label = y_test[i]
                pixels = x_test[i].flatten()
                line = f"{label}," + ",".join(str(p) for p in pixels) + "\n"
                f.write(line)
                if (i + 1) % 2000 == 0:
                    print(f"  Processed {i + 1}/{len(x_test)} test samples")
        
        print("Conversion complete using Keras method!")
        return True
    except ImportError:
        print("TensorFlow not available for Keras method")
        return False
    except Exception as e:
        print(f"Error with Keras method: {e}")
        return False

def download_and_extract_mnist():
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz", 
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    base_urls = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
        "https://github.com/cvdfoundation/mnist/raw/main/",
    ]
    
    for file in files:
        extracted_file = file[:-3]
        
        if not os.path.exists(extracted_file):
            print(f"Downloading {file}...")
            downloaded = False
            for base_url in base_urls:
                try:
                    print(f"  Trying {base_url}{file}")
                    urllib.request.urlretrieve(base_url + file, file)
                    downloaded = True
                    break
                except Exception as e:
                    print(f"  Failed: {e}")
                    continue
            if not downloaded:
                print(f"Could not download {file} from any source!")
                print("\nManual download instructions:")
                print("1. Go to http://yann.lecun.com/exdb/mnist/")
                print("2. Download these files manually:")
                print("   - train-images-idx3-ubyte.gz")
                print("   - train-labels-idx1-ubyte.gz") 
                print("   - t10k-images-idx3-ubyte.gz")
                print("   - t10k-labels-idx1-ubyte.gz")
                print("3. Extract them using gunzip or any archive tool")
                print("4. Run the converter again")
                return False
            
            try:
                print(f"Extracting {file}...")
                with gzip.open(file, 'rb') as f_in:
                    with open(extracted_file, 'wb') as f_out:
                        f_out.write(f_in.read())
                os.remove(file)
                print(f"Successfully extracted {extracted_file}")
            #nahi hua so exception 
            except Exception as e:
                print(f"Error extracting {file}: {e}")
                return False
        else:
            print(f"{extracted_file} already exists, skipping download")
    
    return True

def convert(imgf, labelf, outf, n):
    try:
        f = open(imgf, "rb")
        o = open(outf, "w")
        l = open(labelf, "rb")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the MNIST files are downloaded first!")
        return False
    f.read(16)
    l.read(8)
    images = []

    print(f"Converting {n} samples from {imgf} and {labelf} to {outf}...")

    for i in range(n):
        image = [ord(l.read(1))] 
        for j in range(28*28):  
            image.append(ord(f.read(1)))
        images.append(image)
        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1}/{n} samples")
    for image in images:
        o.write(",".join(str(pix) for pix in image) + "\n")
    
    f.close()
    o.close()
    l.close()
    print(f"Conversion complete! Saved to {outf}")
    return True
print("Checking for MNIST files...")
success = False
if download_and_extract_mnist():
    print("\nStarting IDX to CSV conversion...")
    if convert("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
               "mnist_train.csv", 60000):
         
        if convert("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte",
                  "mnist_test.csv", 10000):
            success = True

if not success:
    print("\nAutomatic download failed. Trying alternative method...")
    if download_mnist_via_keras():
        success = True

if success:
    print("\nAll conversions completed successfully!")
    print("Files created:")
    print("- mnist_train.csv (60,000 samples)")
    print("- mnist_test.csv (10,000 samples)")
    print("\nYou can now use these files with your TensorFlow training script!")
else:
    print("\nBoth automatic methods failed.")
    print("\nManual download option:")
    print("1. Visit: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv")
    print("2. Download 'mnist_train.csv' and 'mnist_test.csv'")
    print("3. Use them directly with your training script")
