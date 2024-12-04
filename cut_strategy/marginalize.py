import pandas as pd


class Marginalizer:
    """
    Encapsulates functionality for marginalizing probability tables.
    """

    @staticmethod
    def get_marginalize_channel(table_prob, current_channels, all_channels="ABC"):
        """
        Marginalizes a probability table by removing specified channels.

        Args:
            table_prob (pd.DataFrame): Probability table to marginalize.
            current_channels (str): Channels to retain.
            all_channels (str): All channels present in the table.

        Returns:
            pd.DataFrame: Marginalized probability table.
        """
        table_prob = table_prob.copy()
        table_prob["state"] = table_prob.index
        new_channels = all_channels

        for channel in all_channels:
            if channel not in current_channels:
                table_prob, new_channels = Marginalizer.marginalize_table(
                    table_prob, channel, new_channels
                )

        table_prob = table_prob.reset_index().set_index("state")
        table_prob.index.name = None
        table_prob = table_prob.drop(columns=["index"], errors="ignore")

        return table_prob

    @staticmethod
    def marginalize_table(table, channel, channels="ABC"):
        """
        Removes a channel from the state column and averages the table over new states.

        Args:
            table (pd.DataFrame): Probability table to marginalize.
            channel (str): Channel to remove.
            channels (str): All channels in the table.

        Returns:
            tuple: (Marginalized table, Updated channels).
        """
        position_element = channels.find(channel)
        table["state"] = table["state"].apply(
            Marginalizer.modify_state, element=position_element
        )
        marginalized_table = table.groupby("state").mean().reset_index()
        new_channels = Marginalizer.change_channels(channel, channels)

        return marginalized_table, new_channels

    @staticmethod
    def modify_state(state, element):
        """
        Modifies a state by removing a specific channel position.

        Args:
            state (str): Original state.
            element (int): Position of the channel to remove.

        Returns:
            str: Modified state.
        """
        state = list(state)
        del state[element]
        return "".join(state)

    @staticmethod
    def change_channels(channel, channels="ABC"):
        """
        Updates the list of channels after a marginalization.

        Args:
            channel (str): Channel to remove.
            channels (str): All channels in the table.

        Returns:
            str: Updated list of channels.
        """
        element = channels.find(channel)
        if element != -1:
            return channels[:element] + channels[element + 1 :]

        return channels
