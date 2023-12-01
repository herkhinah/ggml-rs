use proc_macro2::{Literal, Span, TokenStream, TokenTree};
use quote::{quote, ToTokens};
use syn::{
    parse_macro_input,
    punctuated::Punctuated,
    Attribute, Data, DeriveInput, Field, GenericArgument, Ident,
    Meta::{self},
    MetaList, Path, PathArguments, Type, TypePath, Lifetime, GenericParam, token::Comma,
};

#[derive(Debug)]
enum ParsedType {
    Vec(Box<ParsedType>),
    Tensor(Option<TensorInitMode>),
    Struct(TypePath),
}

#[derive(Debug)]
struct ParsedStruct {
    ident: Ident,
    ty: Path,
    fields: Vec<ParsedStructField>,
}

impl ParsedStructField {
    fn escaped_ident(&self) -> Ident {
        Ident::new(
            format!("__{}", self.ident.to_string()).as_str(),
            Span::call_site(),
        )
    }
}

#[derive(Debug)]
struct ParsedStructField {
    ident: Ident,
    name: Literal,
    ty: ParsedType,
    field: Field
}

#[derive(Debug)]
enum TensorInitMode {
    Load,
}

macro_rules! format_err {
    ($span:expr, $msg:expr $(,)?) => {
        syn::Error::new_spanned(&$span as &dyn quote::ToTokens, &$msg as &dyn std::fmt::Display)
    };
    ($span:expr, $($tt:tt)*) => {
        format_err!($span, format!($($tt)*))
    };
}

macro_rules! bail {
    ($($tt:tt)*) => {
        return Err(format_err!($($tt)*))
    };
}

impl TryFrom<Type> for ParsedType {
    type Error = syn::Error;

    fn try_from(value: Type) -> Result<Self, Self::Error> {
        match value {
            Type::Path(TypePath { qself, path }) => {
                let Some(ty) = path.segments.last() else {
                    bail!(path, "invalid type path");
                };

                // todo: don't assume std::vec::Vec
                if ty.ident.to_string() == "Vec" {
                    let PathArguments::AngleBracketed(args) = &ty.arguments else {
                        panic!();
                    };

                    let GenericArgument::Type(ty) = args.args.first().unwrap() else {
                        bail!(ty, "missing type argument for Vec");
                    };

                    let ty = ParsedType::try_from(ty.clone())?;

                    return Ok(ParsedType::Vec(Box::new(ty)));
                }

                if ty.ident.to_string() == "GTensor" {
                    return Ok(ParsedType::Tensor(None));
                }

                Ok(ParsedType::Struct(TypePath { qself, path: path.clone()} ))
            }
            ty => bail!(ty, "unexpected type"),
        }
    }
}

impl TryFrom<Field> for ParsedStructField {
    type Error = syn::Error;

    fn try_from(value: Field) -> Result<Self, Self::Error> {
        let Field {
            attrs,
            ident,
            ty, ..
        } = value.clone();
        let mut ty = ParsedType::try_from(ty)?;

        if let ParsedType::Tensor(init_mode) = &mut ty {
            *init_mode = Some(TensorInitMode::Load);
        }

        let Some(ident) = ident else {
            bail!(value, "struct field has no identifier");
        };

        let name = gguf_path_segment(attrs.as_slice())?
            .unwrap_or_else(|| Literal::string(ident.to_string().as_str()));

        Ok(ParsedStructField { ident, name, ty, field: value })
    }
}

impl TryFrom<DeriveInput> for ParsedStruct {
    type Error = syn::Error;

    fn try_from(value: DeriveInput) -> Result<Self, Self::Error> {
        let DeriveInput {
            ident,
            data, ..
        } = value.clone();
        let orig = match data {
            Data::Struct(data) => data,
            Data::Enum(_) => bail!(value, "expected struct got enum"),
            Data::Union(_) => bail!(value, "expected struct got union"),
        };

        let fields = orig.fields
            .iter()
            .cloned()
            .map(|f| ParsedStructField::try_from(f))
            .collect::<Result<Vec<_>, syn::Error>>()?;
        let ty = Path::from(ident.clone());

        Ok(ParsedStruct { ident, ty, fields })
    }
}


fn gguf_path_segment(attrs: &[Attribute]) -> syn::Result<Option<Literal>> {
    let mut path_segment = None;

    for attrs in attrs {
        let Meta::List(MetaList { path, tokens, .. }) = &attrs.meta else {
            continue;
        };

        let Some(_) = path.segments.last().map(|s| s.ident.to_string()) else {
            continue;
        };

        for token in tokens.into_token_stream() {
            let TokenTree::Literal(literal) = token.clone() else {
                bail!(token, "expected string literal");
            };

            if path_segment.is_some() {
                bail!(token, "duplicate attribute");
            }

            path_segment = Some(literal);
        }
    }
    Ok(path_segment)
}

fn build_struct_field_vec(
    ident: &Ident,
    name: &Literal,
    ty: &ParsedType,
    index: String,
    format_str: String,
) -> TokenStream {
    let format_str = format!("{format_str}.{{{index}}}");
    let f_expr = {
        let format_str = Literal::string(format_str.as_str());
        quote! {
            format!(#format_str, #name)
        }
    };

    match ty {
        ParsedType::Vec(ty) => {
            let mut inner_index = index.clone();
            inner_index.push_str("_");
            let inner_ident = Ident::new("inner_vec", Span::call_site());

            let inner_loop =
                build_struct_field_vec(&inner_ident, name, ty, inner_index, format_str);
            let index = Ident::new(index.as_str(), Span::call_site());

            quote! {
                let #ident = {
                    let mut vec = ::std::vec::Vec::new();
                    for #index in (0..) {
                        #inner_loop
                        vec.push(#inner_ident);
                    }
                    vec
                };
            }
        }
        ParsedType::Tensor(_) => {
            let index = Ident::new(index.as_str(), Span::call_site());
            quote! {
                let #ident = {
                    let mut vec = ::std::vec::Vec::new();
                    let mut #index = 0;
                    while let Some(tensor) = builder.tensor_info(#f_expr.as_str()) {
                        vec.push(unsafe { GTensor::null() });
                        #index += 1;
                        tensors += 1;
                    }
                    vec
                };
            }
        }
        ParsedType::Struct(_) => {
            let index = Ident::new(index.as_str(), Span::call_site());
          
            quote! {
                let #ident = {
                    let mut vec = ::std::vec::Vec::new();
                    let mut #index = 0usize;
                    while let Ok((value, tensors_)) = ggml::gguf::Deserialize::deserialize_relative(#f_expr, builder) {
                        vec.push(value);
                        #index += 1;
                        tensors += tensors_;
                    }
                    vec
                };
            }
        }
    }
}

fn build_struct_fields(pstruct: &ParsedStruct) -> TokenStream {
    let ParsedStruct {  ty, fields , ..} = pstruct;

    let initialize_fields = TokenStream::from_iter(fields
        .iter()
        .map(|f @ ParsedStructField {  name, ty, .. }| {
            let escaped_ident = f.escaped_ident();

            match ty {
                ParsedType::Vec(ty) => build_struct_field_vec(
                    &escaped_ident,
                    name,
                    ty,
                    "i".to_owned(),
                    "{root}.{}".to_owned(),
                ),
                ParsedType::Tensor(_) => {
                    quote! {
                        let #escaped_ident = unsafe { GTensor::null() };
                        tensors += 1;
                    }
                }
                ParsedType::Struct(path) => {
                   
                    quote! {
                        let #escaped_ident: #path = {
                            let (#escaped_ident, tensors_) = ggml::gguf::Deserialize::deserialize_relative(format!("{root}.{}", #name).as_str(), builder)?;
                            tensors += tensors_;
                            #escaped_ident
                        }
                    }
                },
            }
        }));

    

    let assemble_struct_fields = TokenStream::from_iter(fields.iter().map(|f|  {
        let ident = f.ident.clone();
        let eident = f.escaped_ident();

        quote! { #ident: #eident, }
    }));

    quote! {
        #initialize_fields
        Ok((#ty {
            #assemble_struct_fields
        }, tensors))
    }    

}

fn register_tensors_vec(
    
    ident: &Ident,
    ty: &ParsedType,
    index: String,
    format_str: String,
    
) -> TokenStream {

    let index = Ident::new(index.as_str(), Span::call_site());
    
    match ty {
        ParsedType::Vec(ty) => {
            let ident2 = Ident::new("value", Span::call_site());
            let mut index2 = index.to_string();
            index2.push_str("_");
            let mut format_str = format_str;
            format_str.push_str(".{");
            format_str.push_str(index2.as_str());
            format_str.push_str("}");

            let inner_loop = register_tensors_vec(&ident2,  ty, index2, format_str);

            quote! {
                let mut #ident = unsafe { &*(#ident.as_ptr() as *mut Vec<ghost_cell::GhostCell< '_, _  >>) };
                for (#index, value) in #ident.iter().enumerate() {
                    #inner_loop
                }
            }

        },
        ParsedType::Tensor(_) => quote! {
            let mut #ident = unsafe { &*(#ident.as_ptr() as *mut Vec<ghost_cell::GhostCell< '_, GTensor<'_>  >>) };
            for (#index, value) in #ident.iter().enumerate() {
                *value.borrow_mut(token) = builder.load_tensor(format!(#format_str).as_str())?;
            }
        },
        ParsedType::Struct(ty) => {
            quote! {
                let mut #ident = unsafe { &*(#ident.as_ptr() as *mut Vec<ghost_cell::GhostCell< '_, #ty  >>) };
                
                for (#index, value) in #ident.iter().enumerate() {
                    ggml::gguf::Deserialize::register_tensors(value, token, format!(#format_str), builder);
                }
            }
        },
    }
}

fn destruct_ghost_fields(pstruct: &ParsedStruct) -> TokenStream {
    let ParsedStruct { fields , ..} = pstruct;

    let fields = fields.iter().map(|t| { let ident = &t.ident; let eident = t.escaped_ident(); quote! { #ident: #eident, }});

    TokenStream::from_iter(fields)
  
}

fn register_tensors(pstruct: &ParsedStruct) -> TokenStream {
    let ParsedStruct { fields , ..} = pstruct;

    let fields = fields.iter().map(|f| {
        let ParsedStructField {
            name,
            ty,
            ..
        } = f;

        let eident = f.escaped_ident();

        match ty {
            ParsedType::Vec(ty) =>  register_tensors_vec(&eident, ty, "i".into(), "{root}.{i}".into()),
            ParsedType::Tensor(None) => todo!(),
            ParsedType::Tensor(Some(init)) => match init {
                TensorInitMode::Load => {
                    quote! { *#eident.borrow_mut(token) = builder.load_tensor(#name)?; }
                },
            },
            ParsedType::Struct(_) => {
                quote! { ggml::gguf::Deserialize::register_tensors(#eident, token, format!("{root}.#name"), builder);  }
            },
        }
    });

    TokenStream::from_iter(fields)
    
}

#[proc_macro_derive(Deserialize, attributes( rename ))]
pub fn derive_deserialize_fn(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let pstruct = match ParsedStruct::try_from(ast.clone()) {
        Ok(pstruct) => pstruct,
        Err(e) => return e.into_compile_error().into(),
    };

    let ParsedStruct { ident,  ..} = &pstruct;

    let build_struct = build_struct_fields(&pstruct);
    let destructure_fields = destruct_ghost_fields(&pstruct);
    let register_tensors = register_tensors(&pstruct);
    let ghost_cell_struct_ident = Ident::new(format!("__GhostCellProject_{}", ident.to_string()).as_str(), Span::call_site());
    let gc_project = gc_project(ident, &ast.generics.params, &pstruct.fields);


    let expanded = quote! {
        #gc_project
        
        impl<'a> Deserialize<'a> for #ident <'a> {
            fn deserialize_relative<B: ggml::builder::Builder<'a>>(root: String, builder: &B) -> Result<(Self, usize), B::Error> {
                let mut tensors = 0;
                #build_struct
            }

            fn register_tensors<'id, B: ggml::builder::Builder<'a>>(
                this: &ghost_cell::GhostCell<'id, Self>,
                token: &mut ghost_cell::GhostToken<'id>,
                root: String,
                builder: &mut B
            ) -> Result<(), B::Error> {
                let #ghost_cell_struct_ident {
                    #destructure_fields
                } = this.as_struct_of_cells();

                #register_tensors

                Ok(())

            }
        }

    };

    proc_macro::TokenStream::from(expanded)
}

fn gc_project(ident: &Ident, params: &Punctuated<GenericParam, Comma>, fields: &[ParsedStructField]) -> TokenStream {
    let project_struct =      Ident::new(
            format!("__GhostCellProject_{}", &ident.to_string()).as_str(),
            Span::call_site(),
        );

    let project_trait = Ident::new(
        format!("__GhostCellProjectTrait_{}", &ident.to_string()).as_str(),
        Span::call_site(),
    );
    


    let brand = Lifetime::new("'id", Span::call_site());
    let struct_fields = fields.into_iter().map( |f| {
        let Field {
             attrs,
             vis,
             mutability,
             ident,
             colon_token,
             ty,
         } = f.field.clone();
        let ty = quote! {
            ghost_cell::ghost_cell::GhostCell< #brand, #ty >
        };

        let field = Field {
            ty: syn::parse2(ty).unwrap(),
            attrs: Vec::new(),
            vis,
            mutability,
            ident,
            colon_token,
        };

        quote! { #field, }
    });

    let struct_fields = TokenStream::from_iter(struct_fields);

    quote! {

        struct #project_struct < #brand, #params > {
            #struct_fields

        }

        pub trait #project_trait {
            type StructOfCells;

            fn as_struct_of_cells(&self) -> &Self::StructOfCells;
        }

        impl < #brand, #params > #project_trait for ::ghost_cell::GhostCell< #brand, #ident < #params >> {
            type StructOfCells = #project_struct < #brand, #params >;

            fn as_struct_of_cells(&self) -> &Self::StructOfCells {
                unsafe { &*(self.as_ptr() as *mut #project_struct < #brand, #params >) }
            }


        }
    }
}
