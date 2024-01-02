use proc_macro2::{Literal, Span, TokenStream, TokenTree};
use quote::{quote, ToTokens};
use syn::{
    Attribute, Data, DeriveInput, Field, GenericArgument, Ident,
    Meta::{self},
    MetaList, Path, PathArguments, Type, TypePath, parse_macro_input,
};

#[derive(Debug)]
enum ParsedType {
    Vec(Box<ParsedType>),
    Tensor(Option<TensorInitMode>),
    Struct(TypePath),
    KeyValue,
    KeyValueVec
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
}

#[derive(Debug)]
enum TensorInitMode {
    Load,
    Skip
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
                let res = match ty.ident.to_string().as_str() {
                "Vec" => {
                    let PathArguments::AngleBracketed(args) = &ty.arguments else {
                        panic!();
                    };

                    let GenericArgument::Type(ty) = args.args.first().unwrap() else {
                        bail!(ty, "missing type argument for Vec");
                    };

                    let ty = ParsedType::try_from(ty.clone())?;

                    if matches!(ty, ParsedType::KeyValue) {
                        ParsedType::KeyValueVec
                    } else {
                        ParsedType::Vec(Box::new(ty))
                    }
                }

                "Tensor" => {
                    ParsedType::Tensor(None)
                },
                "u8" | "i8" | "u16" | "i16" | "u32" | "i32" | "u64" | "i64" | "f32" | "f64" | "String" | "str" => ParsedType::KeyValue,
                _ => ParsedType::Struct(TypePath { qself, path: path.clone()} )
            };

            return Ok(res);

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
        let attrs = ParsedAttributes::try_from(attrs.as_slice())?;
        
        if let ParsedType::Tensor(init_mode) = &mut ty {
            if attrs.skip {
                *init_mode = Some(TensorInitMode::Skip);
            } else {
                *init_mode = Some(TensorInitMode::Load);
            }
        }

        let Some(ident) = ident else {
            bail!(value, "struct field has no identifier");
        };


        let name = attrs.key
            .unwrap_or_else(|| {
                let mut ident = ident.to_string();
                if matches!(ty, ParsedType::Tensor(_)) {
                    ident.push_str(".weight");
                }
                Literal::string(ident.as_str())
            });

        Ok(ParsedStructField { ident, name, ty, })
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


struct ParsedAttributes {
    key: Option<Literal>,
    skip: bool
}

impl Default for ParsedAttributes {
    fn default() -> Self {
        Self {
            key: None,
            skip: false
        }
    }
}

impl TryFrom<&[Attribute]> for ParsedAttributes {
    type Error = syn::Error;

    fn try_from(attrs: &[Attribute]) -> Result<Self, Self::Error> {
        let mut path_segment = ParsedAttributes::default();
        for attrs in attrs {
            let Meta::List(MetaList { path, tokens, .. }) = &attrs.meta else {
                continue;
            };

            if let Some(attr) = path.segments.last().map(|s| s.ident.to_string()) {
                match attr.as_str() {
                    "skip" => { path_segment.skip = true; }
                    "rename" => {
                        for token in tokens.into_token_stream() {
                            let TokenTree::Literal(literal) = token.clone() else {
                                bail!(token, "expected string literal");
                            };

                            if path_segment.key.is_some() {
                                bail!(token, "duplicate attribute");
                            }

                            path_segment.key = Some(literal);
                        }
                    }
                    _ => {}
                }
                
                continue;
            };

        }
        Ok(path_segment)
        
    }
}


fn build_struct_field_vec(
    ident: &Ident,
    name: &Literal,
    ty: &ParsedType,
    index: String,
    format_str: String,
) -> TokenStream {
    let format_str = format!("{format_str}{{}}.");
    let f_expr = {
        let format_str = Literal::string(format_str.as_str());
        let index = Literal::string(index.as_str());
        quote! {
            ::std::fmt::format(format_args!(#format_str, #name, #index))
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
                    let mut #index = 0;
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
                    while let Ok(tensor) = builder.get_tensor_info(#f_expr.as_str()) {
                        vec.push(unsafe { Tensor::null() });
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
       ParsedType::KeyValueVec => {
            let index = Ident::new(index.as_str(), Span::call_site());
          
            quote! {
                let #ident = {
                    let mut vec = ::std::vec::Vec::new();
                    let mut #index = 0usize;
                    while let Ok((value)) = builder.get_key_value(std::fmt::format(format_args!("{root}{}", #name)).as_str()) {
                        vec.push(value.try_into()?);
                        #index += 1;
                    }
                    vec
                };
            }
            
        }
       ParsedType::KeyValue => {
            panic!("impossible: please report bug")
       }
    }
}


fn check_tensors(pstruct: &ParsedStruct) -> TokenStream {
    let ParsedStruct { fields , ..} = pstruct;
    let check_tensors = TokenStream::from_iter(fields.iter().filter(|f| matches!(f.ty, ParsedType::Tensor(_))).map(|f| {
        let ParsedStructField { name, .. } = f;
        quote! {
            builder.get_tensor_info(::std::fmt::format(format_args!("{root}{}", #name)).as_str())?;
            
        }
        
    }));

    check_tensors
    
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
                    "{root}{}.".to_owned(),
                ),
                ParsedType::Tensor(_) => {
                    quote! {
                        let #escaped_ident = unsafe { Tensor::null() };
                        tensors += 1;
                    }
                }
                ParsedType::Struct(path) => {
                   
                    quote! {
                        let #escaped_ident: #path = {
                            let (#escaped_ident, tensors_) = ggml::gguf::Deserialize::deserialize_relative(
                                std::fmt::format(format_args!("{root}{}.", #name)), builder
                            )?;
                            tensors += tensors_;
                            #escaped_ident
                        };
                    }
                },
                ParsedType::KeyValueVec | ParsedType::KeyValue => {
                    quote! {
                        let #escaped_ident = builder.get_key_value(std::fmt::format(format_args!("{root}{}", #name)).as_str())?.try_into()?;
                    }
                }
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
            let inner_loop = {
                let ident = Ident::new("value", Span::call_site());
                let index = format!("_{index}");
                let format_str= format!("{format_str}.{{{index}}}" );
                register_tensors_vec(&ident,  ty, index, format_str)
            };

            quote! {
                for (#index, value) in #ident.iter_mut().enumerate() {
                    #inner_loop
                }
            }

        },
        ParsedType::Tensor(_) => quote! {
            for (#index, value) in #ident.iter_mut().enumerate() {
                *value = builder.load_tensor(format!(#format_str).as_str())?;
            }
        },
        ParsedType::Struct(_) => {
            quote! {
              
                for (#index, value) in #ident.iter_mut().enumerate() {
                    value.register_tensors(format!(#format_str), builder);
                }
            }
        },
        ParsedType::KeyValue | ParsedType::KeyValueVec => TokenStream::new()
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
            ParsedType::Vec(ty) =>  register_tensors_vec(&eident, ty, "i".into(), "{root}{i}".into()),
            ParsedType::Tensor(None) => todo!(),
            ParsedType::Tensor(Some(init)) => match init {
                TensorInitMode::Load => {
                    quote! { *#eident = builder.load_tensor(#name)?; }
                },
                TensorInitMode::Skip => quote! {},
            },
            ParsedType::Struct(_) => {
                quote! { #eident.register_tensors( format!("{root}{}", #name), builder);  }
            },
            ParsedType::KeyValue | ParsedType::KeyValueVec => TokenStream::new()
        }
    });

    TokenStream::from_iter(fields)
    
}

#[proc_macro_derive(Deserialize, attributes(rename,skip))]
pub fn derive_deserialize_fn(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let pstruct = match ParsedStruct::try_from(ast.clone()) {
        Ok(pstruct) => pstruct,
        Err(e) => return e.into_compile_error().into(),
    };

    //eprintln!("{pstruct:#?}");

    let ParsedStruct { ident, ..  } = &pstruct;


    let args = if ast.generics.params.is_empty() {
        TokenStream::new()
        
    } else {
        quote! {
            <'a>
        }
            
    };

    let check_tensors = check_tensors(&pstruct);
    let build_struct = build_struct_fields(&pstruct);
    let destructure_fields = destruct_ghost_fields(&pstruct);
    let register_tensors = register_tensors(&pstruct);


    let expanded = quote! {
        impl<'a> Deserialize<'a> for #ident #args {
            fn deserialize_relative<B: ggml::builder::Builder<'a>>(root: String, builder: &B) -> Result<(Self, usize), B::Error> {
                let mut tensors = 0;
                #check_tensors
                #build_struct
            }

            fn register_tensors<'id, B: ggml::builder::Builder<'a>>(
                &mut self,
                root: String,
                builder: &mut B
            ) -> Result<(), B::Error> {
                let Self {
                    #destructure_fields
                } = self;

                #register_tensors

                Ok(())
            }
        }
    };

    proc_macro::TokenStream::from(expanded)
}
