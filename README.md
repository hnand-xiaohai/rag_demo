# rag_demo
ai开发学习过程记录


# 入门项目说明：
## 1.在src目录下手动建立db、model两个目录
  db目录用于chroma数据库的存放
  model目录用于bgr、m3e-base本地模型的存放（也可以在config文件里面配置模型路径）

## 手动embedding
  项目只能全量embedding，在data目录下存放知识库文档，运行build_knowledge_base文件向量化文件

## 启动项目
  执行完成build_knowledge_base文件后可以直接运行app文件即可

