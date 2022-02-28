from stellargraph import StellarGraph
from stellargraph.mapper import GraphSAGENodeGenerator
import utils
import tensorflow

def infer(embedding_model, vertices_df, edges_df, batch_size, num_samples):
    hold_out_graph = StellarGraph(vertices_df, edges_df, edge_type_default='follows', node_type_default='user')
    hold_out_gen = GraphSAGENodeGenerator(hold_out_graph, batch_size, num_samples)
    hold_out_gen = hold_out_gen.flow(vertices_df.index)
    if type(embedding_model) in [str]:
        embedding_model = tensorflow.keras.models.load_model(embedding_model)
    emb = embedding_model.predict(hold_out_gen)
    return emb

if __name__ == '__main__':
    batch_size = 100
    num_samples = [15, 10]
    embedding_model_path = 'users_embedding_100_15_10_model'
    edges_df = utils.read_pickle_from_file('test_edges.pkl')
    vertices_df = utils.read_pickle_from_file('test_vertices.pkl')
    vertices_df.drop(['id'], inplace=True, axis=1)
    embeddings = infer(embedding_model_path, vertices_df, edges_df, batch_size, num_samples)