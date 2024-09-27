import os
import matplotlib.pyplot as plt

def visualization(df_viz):
    save_path='/data/visuals/'
    # Plotting the line chart per warehouse
    warehouses = df_viz['warehouse'].unique()

    for warehouse in warehouses:
        df_warehouse = df_viz[df_viz['warehouse'] == warehouse]
        
        base_dir = os.path.dirname(__file__)
        base_dir = os.path.dirname(base_dir)
        file_path = os.path.join(base_dir, f"data\Visuals\{warehouse}_forecasts.png")
        
        plt.figure(figsize=(14, 7))
        plt.plot(df_warehouse['date'], df_warehouse['orders'], label='Actual Orders')
        plt.plot(df_warehouse['date'], df_warehouse['preds'], label='Predicted Orders')
        plt.xlabel('Date')
        plt.ylabel('Orders')
        plt.title(f'Actual vs Predicted Orders for Warehouse {warehouse}')
        plt.legend()
        plt.grid(True)
        plt.xticks(df_warehouse['date'], rotation=45)

        plt.tight_layout()
        plt.savefig(file_path)
        
        