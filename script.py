import psutil
import cpuinfo
import platform
from datetime import datetime
import tensorflow as tf
import numpy as np
def generate_table(data_dict):
    rows = ""
    for key, value in data_dict.items():
        rows += f"<tr><th>{key}</th><td>{value}</td></tr>\n"
    return rows

def generate_html_report(cpu_info, cpu_temp, cpu_freq, cpu_cycles):
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CPU Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
            }}
            h1 {{
                color: #2E86C1;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            table, th, td {{
                border: 1px solid #ddd;
            }}
            th, td {{
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <h1>CPU Report</h1>
        <p>Report generated on: {report_time}</p>

        <h2>CPU Information</h2>
        <table>
            {generate_table(cpu_info)}
        </table>

        <h2>CPU Temperature</h2>
        <table>
            {generate_table(cpu_temp)}
        </table>

        <h2>CPU Frequency</h2>
        <table>
            {generate_table(cpu_freq)}
        </table>

        <h2>CPU Cycles</h2>
        <table>
            {generate_table(cpu_cycles)}
        </table>
    </body>
    </html>
    """
    with open("cpu_report.html", "w") as f:
        f.write(html_template)
    print("HTML report generated as 'cpu_report.html'.")
def get_cpu_info():
    cpu_info = cpuinfo.get_cpu_info()
    return {
        "Processor": cpu_info['brand_raw'],
        "Architecture": cpu_info['arch'],
        "Cores": psutil.cpu_count(logical=False),
        "Logical Processors": psutil.cpu_count(logical=True),
        "Base Frequency": cpu_info['hz_advertised_friendly']
    }

def get_cpu_temperature():
    temp_data = {}
    if platform.system() == "Linux":
        temp = psutil.sensors_temperatures()
        if "coretemp" in temp:
            for entry in temp["coretemp"]:
                temp_data[entry.label] = f"{entry.current}°C"
    else:
        temp_data["Temperature"] = "Not available"
    return temp_data

def get_cpu_frequency():
    freq = psutil.cpu_freq()
    return {
        "Current Frequency": f"{freq.current:.2f} MHz",
        "Min Frequency": f"{freq.min:.2f} MHz",
        "Max Frequency": f"{freq.max:.2f} MHz"
    }

def get_cpu_cycles():
    cpu_cycles = psutil.cpu_times()
    return {
        "User Mode Time": f"{cpu_cycles.user} seconds",
        "System Mode Time": f"{cpu_cycles.system} seconds",
        "Idle Time": f"{cpu_cycles.idle} seconds"
    }

def generate_tensorflow_model(input_data):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, input_shape=(len(input_data),), activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # 3 output categories (High, Moderate, Low performance)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Example dummy training - assuming categories: 0=High, 1=Moderate, 2=Low
    dummy_data = np.array([[3.6, 1, 2], [2.4, 0.5, 1.8], [3.9, 0.7, 2.2]])
    dummy_labels = np.array([0, 1, 2])

    model.fit(dummy_data, dummy_labels, epochs=10, verbose=0)
    return model.predict(np.array([input_data]))  # Predict for the given input data

def suggest_optimizations(predictions):
    categories = ["High Performance", "Moderate Performance", "Low Performance"]
    return f"Optimization Tip: {categories[np.argmax(predictions)]}. Consider tuning system for {categories[np.argmax(predictions)].lower()}."

if __name__ == "__main__":
    cpu_info = get_cpu_info()
    cpu_temp = get_cpu_temperature()
    cpu_freq = get_cpu_frequency()
    cpu_cycles = get_cpu_cycles()

    input_data = [
        float(cpu_freq['Current Frequency'].split()[0]),
        psutil.cpu_count(logical=False),
        psutil.cpu_count(logical=True)
    ]

    predictions = generate_tensorflow_model(input_data)
    optimization_tip = suggest_optimizations(predictions)
    print(optimization_tip)

    
    cpu_info['Optimization Tip'] = optimization_tip
    generate_html_report(cpu_info, cpu_temp, cpu_freq, cpu_cycles)
