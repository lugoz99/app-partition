import pandas as pd


### Dada una tabla de probabilidad y los canales que quedan luego de eliminar uno de ellos
### calcula la marginalizacion de la tabla.
### @ param table_prob: tabla de probabilidad
### @ param current_channels: canales sobrantes
### @ param all_channels: todos los canales de la tabla
def get_marginalize_channel(tabla_marg, current_channels, all_channels='ABC'):
    table_prob = tabla_marg.copy()
    table_prob['state'] = tabla_marg.index
    new_channels = all_channels

    for channel in all_channels:
        if channel not in current_channels:
            table_prob, new_channels = marginalize_table(
                table_prob, channel, new_channels)
            
    table_prob = table_prob.reset_index().set_index('state')
    table_prob.index.name = None
    table_prob = table_prob.drop('index', axis=1)
    if 'state' in table_prob.columns:
        table_prob = table_prob.drop(columns='state', inplace=True)
    
    return table_prob


### Elimina el estado del canal en la posicion del mismo, luego agrupa la tabla
### en los nuevos estados resultantes y promedia sus valores.
def marginalize_table(table, channel, channels='ABC'):
    position_element = channels.find(channel)
    table['state'] = table['state'].apply(modify_state, element=position_element)
    promd_table = table.groupby('state').mean()
    promd_table = promd_table.reset_index()
    new_channels = change_channels(channel, channels)

    return promd_table, new_channels
    
def modify_state(state, element):
    state = list(state)
    del state[element]

    return ''.join(state)


### Retorna los canales sobrantes de la marginalizacion
def change_channels(channel, channels='ABC'):
    element = channels.find(channel)
    if element != -1:
        return channels[:element] + channels[element + 1:]

    return channels
