#![feature(concat_idents)]
use ast::AStruct;
use proc_macro2::{Span, TokenStream};
use proc_macro_common::{bail, format_err};
use quote::quote;
use syn::{
    parse::Parse, parse2, parse_macro_input, token::Colon, Attribute, Data, DataStruct,
    DeriveInput, Field, Fields, FieldsNamed, Ident, Label, Lifetime, Meta, MetaList, Token,
};

use crate::ast::{AField, AType, FieldAttrs, StructAttrs};

mod ast;

enum BranchType {
    Root,
    Node { depth: usize },
}

#[proc_macro]
pub fn fresh(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let mut lifetime: syn::Lifetime = match syn::parse2(input.into()) {
        Ok(l) => l,
        Err(e) => return e.into_compile_error().into(),
    };

    let ident = Ident::new(
        format!("{}_", lifetime.ident.to_string()).as_str(),
        Span::call_site(),
    );
    lifetime.ident = ident;

    quote! {
        #lifetime
    }
    .into()
}

#[proc_macro_attribute]
pub fn gguf(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    //let (attr, item): (_, TokenStream) = (attr, item.into());

    let attr: StructAttrs = match syn::parse(attr.into()) {
        Ok(tokens) => tokens,
        Err(e) => return e.to_compile_error().into(),
    };

    let item_ = item.clone();
    let derive_input: DeriveInput = parse_macro_input!(item);
    let mut orig = match derive_input.clone().data.clone() {
        Data::Struct(data) => data,
        _ => todo!(),
    };

    let fields: Vec<Field> = orig
        .fields
        .clone()
        .into_iter()
        .map(|f| Field {
            attrs: Vec::new(),
            ..f
        })
        .collect();

    orig.fields.iter_mut().for_each(|f| {
        f.attrs = Vec::new();
    });

    let DeriveInput {
        attrs,
        vis,
        ident,
        generics,
        data,
    } = derive_input.clone();
    let fields = orig.fields.into_iter().map(
        |Field {
             attrs,
             vis,
             mutability,
             ident,
             colon_token,
             ty,
         }| {
            let ident = ident.unwrap();
            quote! {
                #vis #ident: #ty,
            }
        },
    );
    let fields = proc_macro2::TokenStream::from_iter(fields);

    let item_ = quote! {
        #vis struct #ident #generics {
            #fields
        }
    };

    let item_: proc_macro::TokenStream = item_.into();

    let astruct = match AStruct::try_from((derive_input, attr)) {
        Ok(astruct) => astruct,
        Err(e) => return e.into_compile_error().into(),
    };

    let impl_block = if astruct.root {
        let init = &astruct.init_macro;
        quote! {
            impl #generics #ident #generics {
                pub fn try_from_gguf(gguf: &GGUFContext) -> Result<Self, GGUFError> {
                    let mut n_tensors: usize = 0;
                    let res = #init!(gguf, n_tensors);
                    res
                }
            }

        }
    } else {
        TokenStream::new()
    };
    let stream: proc_macro::TokenStream = proc_macro::TokenStream::from_iter(
        [item_, construct_struct(astruct).into(), impl_block.into()].into_iter(),
    )
    .into();

    stream
}

fn construct_field(f: &AField, level: usize) -> TokenStream {
    let AField {
        ident,
        escaped_ident,
        format_str,
        ty,
    }: &AField = f;

    let tensor_break_expr = match level {
        0 => quote! {
            return Err(GGUFError::TensorNotFound(tensor_name));
        },
        _ => quote! {
            break $label;

        },
    };

    let struct_break_expr = match level {
        0 => quote! {
            return Err(err);
        },
        _ => quote! {
            break $label;
        },
    };

    let fresh_label = match level {
        0 => {
            let label = Lifetime::new("'_loop", Span::call_site());
            quote! { #label }
        }
        n => {
            quote! { fresh!($label) }
        }
    };

    let mut fresh_index = match level {
        0 => quote! { i__ },
        1 => quote! { i____ },
        _ => quote! { concat_idents!(__, $index)},
    };

    let format_expr = format_str.as_ref().map(|ident| match level {
        0 => quote! {
            #format_str.to_string()
        },
        1 => quote! {
            format!(#format_str, $index)
        },
        _ => quote! {
            format!(#format_str, $index, $($indextt),*)
        },
    });

    let break_expr = if level > 0 {
        quote! {
            if $index == 0 {
                break;
            }
        }
    } else {
        TokenStream::new().into()
    };

    let prev_indices = TokenStream::new();

    match ty {
        ast::AType::Vec(ty) => {
            let format_expr = format_str.as_ref().map(|format_str| match level {
                0 => quote! {
                    format!(#format_str, #fresh_index)
                },
                1 => quote! {
                    format_reverse!(#format_str, #fresh_index, $index)
                },
                _ => quote! {
                    format_reverse!(#format_str, #fresh_index, $index, $($indextt),*)
                },
            });

            match ty.as_ref() {
                ast::AType::Tensor { format_str, dims } => {
                    quote! {
                        {
                            let mut #escaped_ident = Vec::new();
                            let mut #fresh_index = 0usize;
                            let mut tensor_name = #format_expr;
                            while let Ok(_) = builder.get_tensor_info(tensor_name.as_str())
                            {
                                #escaped_ident.push(unsafe { Tensor::null() });
                                #fresh_index += 1;
                                n_tensors += 1;
                                tensor_name = #format_expr;
                            }
                            if #fresh_index == 0 {
                                Err(GGUFError::TensorNotFound(tensor_name))
                            }
                            else {
                                Ok(
                                    #escaped_ident
                                )
                            }
                        }
                    }
                }
                ast::AType::Struct {
                    ty,
                    init_macro,
                    load_macro,
                } => {
                    quote! {

                        {
                            let mut #escaped_ident = Vec::new();
                            let mut res: Result<_, GGUFError>;
                            let mut #fresh_index = 0usize;
                            loop {
                                res = #init_macro!($gguf, $n_tensors, #fresh_index, #prev_indices);
                                match res {
                                    Ok(res) =>  #escaped_ident.push(res),
                                    Err(e) => {
                                        if #escaped_ident.is_empty() {
                                            break Err(e)
                                        } else {
                                            break Ok(#escaped_ident)
                                        }
                                    },
                                }
                                #fresh_index += 1;
                            }
                        }
                    }
                }
                _ => todo!(),
            }
        }
        ast::AType::Tensor { format_str, dims } => {
            quote! {
                {
                    let tensor_name = #format_expr;
                    match $gguf.tensor_infos.get(tensor_name.as_str()) {
                        Some(_) => Ok(unsafe { Tensor::null() }),
                        None => Err(GGUFError::TensorNotFound(tensor_name)),
                    }
                }
            }
        }
        ast::AType::Struct {
            ty,
            init_macro,
            load_macro,
        } => match level {
            0 => quote! { #init_macro($gguf, $n_tensors) },
            1 => quote! { #init_macro($gguf, $n_tensors, $index) },
            _ => quote! { #init_macro($gguf, $n_tensors, $index, $($indextt),*) },
        },
        ast::AType::KeyValue { format_str } => todo!(),
        ast::AType::KeyValueVec { format_str } => todo!(),
    }
}

fn construct_struct(s: AStruct) -> TokenStream {
    let AStruct {
        ident,
        init_macro,
        load_macro,
        ty,
        fields,
        root,
    } = s;

    let assemble_fields = TokenStream::from_iter(fields.iter().map(
        |AField {
             ident,
             escaped_ident,
             ..
         }| quote! { #ident: #escaped_ident,  },
    ));

    let fields0 = TokenStream::from_iter(fields.iter().map(|f| {
        let lv = &f.ident;
        let rv = construct_field(f, 0);
        quote! {
            #lv: #rv?,
        }
    }));
    let fields1 = TokenStream::from_iter(fields.iter().map(|f| {
        let lv = &f.ident;
        let rv = construct_field(f, 1);
        quote! {
            #lv: #rv?,
        }
    }));
    let fields2 = TokenStream::from_iter(fields.iter().map(|f| {
        let lv = &f.ident;
        let rv = construct_field(f, 2);
        quote! {
            #lv: #rv?,
        }
    }));

    let res = quote! {

        #[macro_export]
        macro_rules! #init_macro {
            ($gguf:ident, $n_tensors:ident) => {
                {
                    Ok(#ident { #fields0 })
                }
            };
            ($gguf:ident, $n_tensors:ident, $index:ident) => {
                {
                    Ok(#ident { #fields1 })
                }
            };
            ($gguf:ident, $n_tensors:ident, $index:ident, $($indextt:ident),*) => {
                {
                    Ok(#ident { #fields2 })
                }
            };
        }
    };
    eprintln!("{}", res.to_string());

    res
}

/*
macro_rules! construct_struct {
    ($n_tensors:ident, $break_expr:tt) => {
        $labela: while true {
            $labela: while true {
                break $labela;

            }
            break $labela;
        }
    };
    ($n_tensors:ident,  $break_expr:tt, $index:ident $(, $indextt:ident)*) => {
        $labela: while true {
            $labela: while true {
                break $labela;

            }
            $break_expr
        }
    }
}

fn test() {
    for i in (0..).into_iter() {
        for j in (0..).into_iter() {
            loop_tensor!(foobar, "tensor.{0}.{1}", i, j);
        }
    }
}
*/
