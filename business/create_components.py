from loko_extensions.model.components import Arg, Component, save_extensions, Input, Output, AsyncSelect, Dynamic

# from model.components import Input, Arg, Output, Component, save_extensions, Select, AsyncSelect, Dynamic

########################### CRUDS COMPONENT ################################
new_collection_name = Arg(name="new_collection_name",
                          type="text",
                          label="New Images Collection Name",
                          helper="Helper text",
                          group="Create Images Collection")
collection_name = AsyncSelect(name="collection_name",
                              label="Images Collection Name",
                              helper="Helper text",
                              group="Select Collection",
                              url='http://localhost:9999/routes/recommender/collections_')

get_collections = Input(id='get_collections',
                        label='Get Collections',
                        service='collections',
                        to='get_collections')

get_collections_ = Output(id='get_collections',
                          label='Get Collections')

create_collection = Input(id='create_collection',
                          label='Create Collection',
                          service='create_collection',
                          to='create_collection')

create_collection_ = Output(id='create_collection',
                            label='Create Collection')

update_collection = Input(id='update_collection',
                          label='Update Collection',
                          service='update_collection',
                          to='update_collection')

update_collection_ = Output(id='update_collection',
                            label='Update Collection')

delete_collection = Input(id='delete_collection',
                          label='Delete Collection',
                          service='delete_collection', to='delete_collection')

delete_collection_ = Output(id='delete_collection', label='Delete Collection')

comp_cruds = Component(name="CRUD collections",
                       args=[new_collection_name, collection_name],
                       inputs=[get_collections, create_collection, update_collection, delete_collection],
                       outputs=[get_collections_, create_collection_, update_collection_, delete_collection_]
                       )
######################### SEARCH COMPONENT ###############################

max_items = Arg(name="max_items",
                type="text",
                label="max_items",
                helper="Maximum similar items to return",
                required=True,
                value=2)

search = Input(id='input',
               label='Search',
               service='find_similar_images',
               to='item_collection')  # to = id_output

search_ = Output(id='item_collection',
                 label='Search')

search_engine = Component(name="Images Search",
                          args=[collection_name, max_items],
                          inputs=[search],
                          outputs=[search_],
                          configured=False)

#######################################################

###################### RECOMMDER COMPONENT ###############
collection_name_ = AsyncSelect(name="collection_name_",
                               label="Images Collection Name",
                               helper="Helper text",
                               url='http://localhost:9999/routes/recommender/collections_')

create_rec = Input(id='input_create_rec',
                   label='Create',
                   service='recsys/orders',
                   to='output_create_rec')  # to = id_output

create_rec_ = Output(id='output_create_rec',
                     label='Create')

delete_rec = Input(id='delete_rec',
                   label='Delete',
                   service='delete/orders',
                   to='output_create_rec')

delete_rec_ = Output(id='delete_rec',
                     label='Delete')

available_users = Input(id='available_users',
                        label='Users',
                        service='recsys/get_user_id',
                        to='available_users_')

available_users_ = Output(id='available_users_',
                          label='Users')

recommend = Input(id='recommend',
                  label='Recommend',
                  service='recsys/recommend',
                  to='recommend_')

recommend_ = Output(id='recommend_',
                    label='Recommend')

months_training = Arg(name="months_training",
                      type="text",
                      label="Months Training",
                      helper="Number of months to use for training",
                      group="Create Recommender",
                      value=5
                      )
min_item_allowed = Arg(name="min_item_allowed",
                       type="text",
                       label="Minimum Item ",
                       helper="The minimum number of items acceptable to create the recommender",
                       group="Create Recommender",
                       value=2
                       )

user_id = Arg(name="user_id",
              type="text",
              label="User ID",
              helper="Number of months to use for training",
              group="Recommend",
              )


max_items_ = Arg(name="max_items_",
                 type="text",
                 label="max items",
                 helper="Maximum similar items to return",
                 required=True,
                 value=5,
                 group="Recommend",
                 )
recommender = Component(name="Recommender",
                        args=[collection_name_, months_training, min_item_allowed, user_id, max_items_],
                        inputs=[create_rec, delete_rec, available_users, recommend],
                        outputs=[create_rec_, delete_rec_, available_users_, recommend_],
                        configured=False,
                        trigger=True)

#########################################################

save_extensions([comp_cruds, search_engine, recommender])
