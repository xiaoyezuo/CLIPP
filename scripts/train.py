from lib.dataset import TextImagePathDataset

texts = ["This is a test", "This is another test", "This is the final test"]
image_paths = [["/home/zuoxy/ceph_old/navcon_video/rxr_clips/000000/000000.png","/home/zuoxy/ceph_old/navcon_video/rxr_clips/000000/000001.png"],
               ["/home/zuoxy/ceph_old/navcon_video/rxr_clips/000000/000002.png","/home/zuoxy/ceph_old/navcon_video/rxr_clips/000000/000003.png"]]

dataset = TextImagePathDataset(image_paths, texts)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

for text_embed, image_embed in data_loader:
    print("Text Embeddings:", text_embed.shape)  # Should print [batch_size, 512]
    print("Image Embeddings:", image_embed.shape)  # Should print [batch_size, 512]
    similarity = torch.nn.functional.cosine_similarity(text_embed, image_embed, dim=1)
    print("Similarity Scores:", similarity)  # Outputs similarity scores between text and image embeddings for each pair in the batch
